//
// Created by ang on 2021/1/13.
//

#ifndef ONNX2TRT_GATHERELEMENTS_H
#define ONNX2TRT_GATHERELEMENTS_H

#include "serialize.hpp"
#include "NvInferPlugin.h"
#include "gather_elements.cuh"

using namespace nvinfer1;

namespace onnx2trt{
    class GatherElementsPlugin : public nvinfer1::IPluginV2DynamicExt {
        private:
        unsigned int axis_;

        protected:
        void deserialize(void const* data, size_t length);

        void serialize(void *buffer) const override;

        size_t getSerializationSize() const override;

        public:
//        GatherElementsPlugin(unsigned int dim, unsigned int t_c, unsigned int t_h, unsigned t_w, unsigned int idx_c, unsigned int idx_h, unsigned int idx_w);

        explicit GatherElementsPlugin(unsigned int dim);

        GatherElementsPlugin(void const* data, size_t length);

        const char* getPluginType() const override;

        const char* getPluginVersion() const override;

        int getNbOutputs() const override;

//        Dims getOutputDimensions(int index, const Dims *inputs, int n_input_tensors) override;

//        bool supportsFormat(DataType type, PluginFormat format) const override;

        int initialize() override;

        void terminate() override;

//        size_t 	getWorkspaceSize (int32_t maxBatchSize) const override;

        void destroy() override;

        const char *getPluginNamespace() const override;

        void setPluginNamespace(const char *N) override;

//        int enqueue(int batch_size, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;

        // IPluginV2Ext Methods
        DataType getOutputDataType(int index, const DataType *input_types, int n_inputs) const override;

//        bool isOutputBroadcastAcrossBatch(int output_index, const bool *input_is_broadcast, int n_inputs) const override;

//        bool canBroadcastInputAcrossBatch(int input_index) const override;

//        void configurePlugin(const Dims *input_dims, int n_inputs, const Dims *output_dims, int n_outputs,
//                             const DataType *input_types, const DataType *output_types,
//                             const bool *input_is_broadcast, const bool *output_is_broadcast,
//                             PluginFormat float_format, int max_batch_size) override;

        IPluginV2DynamicExt *clone() const override;

        DimsExprs getOutputDimensions (int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) override;

        size_t getWorkspaceSize(const PluginTensorDesc *inputs, int nbInputs, const PluginTensorDesc *outputs,
                                int nbOutputs) const TRTNOEXCEPT override;

        bool supportsFormatCombination (int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) override;

        void configurePlugin (const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) override;

        int32_t enqueue (const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) override;

    };

    class GatherElementsPluginCreator : public IPluginCreator {
    public:
        GatherElementsPluginCreator();

        const char *getPluginName() const override;

        const char *getPluginVersion() const override;

        const char *getPluginNamespace() const override;

        IPluginV2 *deserializePlugin(const char *name, const void *serial_data, size_t serial_length) override;

        void setPluginNamespace(const char *N) override;

        const PluginFieldCollection *getFieldNames() override;

        IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override;
    };

    REGISTER_TENSORRT_PLUGIN(GatherElementsPluginCreator);
}


#endif //ONNX2TRT_GATHERELEMENTS_H
