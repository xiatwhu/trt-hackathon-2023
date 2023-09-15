/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tensorrt_llm/plugins/gemmBiasResPlugin/gemmBiasResPlugin.h"
#include "tensorrt_llm/common/paramsHash.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "cudnn_frontend.h"

#include <chrono>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using nvinfer1::plugin::GemmBiasResPluginCreator;
using nvinfer1::plugin::GemmBiasResPlugin;

static const char* GEMM_BIAS_RES_PLUGIN_VERSION{"1"};
static const char* GEMM_BIAS_RES_PLUGIN_NAME{"GemmBiasRes"};
PluginFieldCollection GemmBiasResPluginCreator::mFC{};
std::vector<PluginField> GemmBiasResPluginCreator::mPluginAttributes;

namespace {

struct CacheKey {
    int M;
    int N;
    int K;
    int has_bias;
};

struct BenchmarkCache {
    std::mutex mutex;
    std::unordered_map<CacheKey, std::shared_ptr<cudnn_frontend::ExecutionPlan>,
                       ParamsHash<CacheKey>, ParamsEqual<CacheKey>> map;

    cudnn_frontend::ExecutionPlan* find(const CacheKey& key) {
        std::lock_guard<std::mutex> guard(mutex);
        auto it = map.find(key);
        if (it != map.end()) {
            return it->second.get();
        }

        return nullptr;
    }

    void insert(const CacheKey& key, cudnn_frontend::ExecutionPlan result) {
        std::lock_guard<std::mutex> guard(mutex);
        map[key] = std::make_shared<cudnn_frontend::ExecutionPlan>(result);
    }
};

BenchmarkCache engine_cache;

cudnn_frontend::Tensor getTensorDescriptor(int row, int col, char id, cudnnDataType_t dataType, bool isVirtual = false) {
#if 0
    int64_t dims[] = {1, row, col};
    int64_t strides[] = {row * col, col, 1};

    auto tensor = cudnn_frontend::TensorBuilder()
                .setDim(3, dims)
                .setStride(3, strides)
                .setId(id)
                .setVirtual(isVirtual)
                .setAlignment(16)
                .setDataType(dataType)
                .build();
#endif
    int64_t dims[] = {row, col, 1, 1};
    int64_t strides[] = {col, 1, col, col};

    auto tensor = cudnn_frontend::TensorBuilder()
                .setDim(4, dims)
                .setStride(4, strides)
                .setId(id)
                .setVirtual(isVirtual)
                .setAlignment(16)
                .setDataType(dataType)
                .build();
    return tensor;
}

bool getEnvValue(const char* key, bool defaultVal) {
    const char* val = getenv(key);
    if (val == nullptr || !*val) {
        return defaultVal;
    }
    bool isTrue = val[0] == 'Y' || val[0] == 'y'        // yes
            || val[0] == 'T' || val[0] == 't'           // true
            || val[0] == '1';                           // 1
    return isTrue;
}

}  // namespace

GemmBiasResPlugin::GemmBiasResPlugin(int hasBias, float beta, nvinfer1::DataType type)
    : mHasBias(hasBias)
    , mBeta(beta)
    , mType(type)
{
}

// Parameterized constructor
GemmBiasResPlugin::GemmBiasResPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mHasBias);
    read(d, mBeta);
    read(d, mType);
    PLUGIN_ASSERT(d == a + length);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* GemmBiasResPlugin::clone() const noexcept
{
    auto* plugin = new GemmBiasResPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
}

nvinfer1::DimsExprs GemmBiasResPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        // A * B + bias + X
        PLUGIN_ASSERT(nbInputs == 3 || nbInputs == 4);
        PLUGIN_ASSERT(outputIndex == 0);

        const int nbDimsX = inputs[nbInputs - 1].nbDims;
        DimsExprs ret;
        ret.nbDims = nbDimsX;

        for (int i = 0; i < nbDimsX; ++i) {
            ret.d[i] = inputs[nbInputs - 1].d[i];
        }

        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool GemmBiasResPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void GemmBiasResPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t GemmBiasResPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return CUDNN_WORKSPACE_SIZE;
}

int GemmBiasResPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // inputs
    //     x [B, T, K]
    //     w [K, N]
    //     b [1, N] (optional)
    //     z [B, T, N]
    // outputs
    //     y [B, T, N]

    cudnnHandle_t cudnnHandle = *mHandle;
    PLUGIN_CUDNNASSERT(cudnnSetStream(cudnnHandle, stream));

    const int nbDimsA = inputDesc[0].dims.nbDims;
    int M = 1;
    const int K = inputDesc[0].dims.d[nbDimsA - 1];
    for (int i = 0; i < nbDimsA - 1; ++i)
    {
        M *= inputDesc[0].dims.d[i];
    }

    const int zIndex = mHasBias ? 3 : 2;
    const int nbDimsZ = inputDesc[zIndex].dims.nbDims;
    const int N = inputDesc[zIndex].dims.d[nbDimsZ - 1];

    CacheKey key;
    key.M = M;
    key.N = N;
    key.K = K;
    key.has_bias = mHasBias;

    int has_bias = mHasBias;
    auto run = [&](cudnn_frontend::ExecutionPlan* plan) -> cudnnStatus_t {
        std::vector<void*> data_ptrs = {const_cast<void*>(inputs[0]),
                                        const_cast<void*>(inputs[1]),
                                        const_cast<void*>(inputs[has_bias ? 3 : 2]),
                                        outputs[0]};
        std::vector<int64_t> uids = {'x', 'w', 'z', 'y'};
        if (has_bias) {
            data_ptrs.push_back(const_cast<void*>(inputs[2]));
            uids.push_back('b');
        }
        auto variantPack = cudnn_frontend::VariantPackBuilder()
                    .setWorkspacePointer(workspace)
                    .setDataPointers(data_ptrs.size(), data_ptrs.data())
                    .setUids(uids.size(), uids.data())
                    .build();
        return cudnnBackendExecute(cudnnHandle, plan->get_raw_desc(), variantPack.get_raw_desc());
    };

    cudnn_frontend::ExecutionPlan* cache_plan = engine_cache.find(key);
    if (cache_plan != nullptr) {
        auto status = run(cache_plan);
        PLUGIN_CUDNNASSERT(status);
        return 0;
    }

    // find best cfg

    auto xTensor = getTensorDescriptor(M, K, 'x', CUDNN_DATA_HALF);
    auto yTensor = getTensorDescriptor(M, N, 'y', CUDNN_DATA_HALF);
    auto wTensor = getTensorDescriptor(N, K, 'w', CUDNN_DATA_HALF);
    auto bTensor = getTensorDescriptor(1, N, 'b', CUDNN_DATA_HALF);
    auto zTensor = getTensorDescriptor(M, N, 'z', CUDNN_DATA_HALF);

    auto afterMatMulTensor = getTensorDescriptor(M, N, 'A', CUDNN_DATA_FLOAT, true);
    auto afterAddTensor = getTensorDescriptor(M, N, 'B', CUDNN_DATA_FLOAT, true);

#if 0
    auto matmulDesc = cudnn_frontend::MatMulDescBuilder()
                .setComputeType(CUDNN_DATA_HALF)
                .build();
#endif
    int64_t pad[] = {0, 0};
    int64_t stride[] = {1, 1};
    int64_t dilation[] = {1, 1};
    auto convDesc = cudnn_frontend::ConvDescBuilder()
                .setComputeType(CUDNN_DATA_FLOAT)
                .setMathMode(CUDNN_CONVOLUTION)
                .setSpatialDimCount(2)
                .setSpatialStride(2, stride)
                .setPrePadding(2, pad)
                .setPostPadding(2, pad)
                .setDilation(2, dilation)
                .build();

    auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_ADD)
                .setMathPrecision(CUDNN_DATA_FLOAT)
                .build();
    auto resDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_ADD)
                .setMathPrecision(CUDNN_DATA_FLOAT)
                .build();

    // std::cout << xTensor.describe() << std::endl;
    // std::cout << wTensor.describe() << std::endl;
    // std::cout << convDesc.describe() << std::endl;
    // std::cout << afterMatMulTensor.describe() << std::endl;
    // std::cout << afterAddTensor.describe() << std::endl;
    auto matmulOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                .setxDesc(xTensor)
                .setwDesc(wTensor)
                .setyDesc(afterMatMulTensor)
                .setcDesc(convDesc)
                .setAlpha(1.0f)
                .setBeta(0.f)
                .build();
    auto resOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(matmulOp.getOutputTensor())
                .setbDesc(zTensor)
                .setyDesc(afterAddTensor)
                .setpwDesc(resDesc)
                .setAlpha(1.f)
                .setAlpha2(mBeta)
                .build();
    auto biasOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(resOp.getOutputTensor())
                .setbDesc(bTensor)
                .setyDesc(yTensor)
                .setpwDesc(biasDesc)
                .setAlpha(1.f)
                .setAlpha2(mBeta)
                .build();

    std::vector<cudnn_frontend::Operation const*> ops = {&matmulOp};
    if (mHasBias) {
        ops.push_back(&biasOp);
    }
    ops.push_back(&resOp);
    auto opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(cudnnHandle)
                .setOperationGraph(ops.size(), ops.data())
                .build();
    
    auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                .setOperationGraph(opGraph)
                .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                .build();

    auto& engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

    float best_time = INFINITY;
    int best_cfg_index = -1;
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    static bool skipFindBest = getEnvValue("SKIP_FIND_BEST", false);
    for (int n = 0; n < engine_config.size(); ++n) {
        auto& cfg = engine_config[n];
        try {
            auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(cudnnHandle)
                    .setEngineConfig(cfg)
                    .build();
            
            auto workspace_size = plan.getWorkspaceSize();
            if (workspace_size > CUDNN_WORKSPACE_SIZE) {
                continue; 
            }

            std::vector<float> times;
            for (int i = 0; i < 11; ++i) {
                float duration_ms;
                cudaEventRecord(start_event, stream);
                auto status = run(&plan);

                cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
                cudaEventRecord(stop_event, stream);
                cudaEventSynchronize(stop_event);
                cudaEventElapsedTime(&duration_ms, start_event, stop_event);
                times.push_back(duration_ms);
            }
            std::sort(times.begin(), times.end());

            // printf("gemmBiasRes: %d %d %d %d %fms\n", M, N, K, n, times[5]);
            if (times.size() > 0 && times[5] < best_time) {
                best_time = times[5];
                best_cfg_index = n;
                engine_cache.insert(key, std::move(plan));

                if (skipFindBest) {
                    break;
                }
            }
        } catch (cudnn_frontend::cudnnException& e) {
            continue;
        }
    }

    PLUGIN_ASSERT(best_cfg_index != -1);

    // std::cout << "best: " << best_cfg_index << std::endl;
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType GemmBiasResPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* GemmBiasResPlugin::getPluginType() const noexcept
{
    return GEMM_BIAS_RES_PLUGIN_NAME;
}

const char* GemmBiasResPlugin::getPluginVersion() const noexcept
{
    return GEMM_BIAS_RES_PLUGIN_VERSION;
}

int GemmBiasResPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int GemmBiasResPlugin::initialize() noexcept
{
    mHandle = getCudnnHandle();
    return 0;
}

void GemmBiasResPlugin::destroy() noexcept
{
    delete this;
}

size_t GemmBiasResPlugin::getSerializationSize() const noexcept
{
    return sizeof(mHasBias) + sizeof(mBeta) + sizeof(mType);
}

void GemmBiasResPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mHasBias);
    write(d, mBeta);
    write(d, mType);
    assert(d == a + getSerializationSize());
}

void GemmBiasResPlugin::terminate() noexcept {}

void GemmBiasResPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* GemmBiasResPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

GemmBiasResPluginCreator::GemmBiasResPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("has_bias", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("beta", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GemmBiasResPluginCreator::getPluginName() const noexcept
{
    return GEMM_BIAS_RES_PLUGIN_NAME;
}

const char* GemmBiasResPluginCreator::getPluginVersion() const noexcept
{
    return GEMM_BIAS_RES_PLUGIN_VERSION;
}

const PluginFieldCollection* GemmBiasResPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* GemmBiasResPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int has_bias = 1;
    float beta = 1.f;
    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "has_bias"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            has_bias = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "beta"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            beta = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new GemmBiasResPlugin(has_bias, beta, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* GemmBiasResPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call GemmBiasResPlugin::destroy()
    try
    {
        auto* obj = new GemmBiasResPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void GemmBiasResPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* GemmBiasResPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
