//
// Created by ang on 2021/1/13.
//

#include "GatherElements.h"
#include <cassert>
#include <stdio.h>

void onnx2trt::GatherElementsPlugin::deserialize(void const *data, size_t length) {
    const char *d = reinterpret_cast<const char *>(data);
    read(d, axis_);
}

void onnx2trt::GatherElementsPlugin::serialize(void *buffer) const {
    char *d = static_cast<char *>(buffer);
    write(d, axis_);
}

size_t onnx2trt::GatherElementsPlugin::getSerializationSize() const {
    return sizeof(axis_);
}


onnx2trt::GatherElementsPlugin::GatherElementsPlugin(
        unsigned int axis) :
        axis_(axis){
}

onnx2trt::GatherElementsPlugin::GatherElementsPlugin(void const *data, size_t length) {
    this->deserialize(data, length);
}

const char *onnx2trt::GatherElementsPlugin::getPluginType() const {
    return "GatherElements";
}

const char *onnx2trt::GatherElementsPlugin::getPluginVersion() const {
    return "1";
}

int onnx2trt::GatherElementsPlugin::getNbOutputs() const {
    return 1;
}

int onnx2trt::GatherElementsPlugin::initialize() { return 0; }

void onnx2trt::GatherElementsPlugin::terminate() {}

void onnx2trt::GatherElementsPlugin::destroy() {
    delete this;
}

const char *onnx2trt::GatherElementsPlugin::getPluginNamespace() const {
    return "";
}

void onnx2trt::GatherElementsPlugin::setPluginNamespace(const char *N) {

}

// IPluginV2Ext Methods
DataType onnx2trt::GatherElementsPlugin::getOutputDataType(int index, const DataType *input_types, int n_inputs) const {
    assert(index == 0 && n_inputs == 2 && input_types[1] == DataType::kINT32);
    return input_types[0];
}

IPluginV2DynamicExt *onnx2trt::GatherElementsPlugin::clone() const {
    return new GatherElementsPlugin(axis_);
}

size_t onnx2trt::GatherElementsPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int nbInputs,
                                                        const PluginTensorDesc *outputs, int nbOutputs) const {
    int n_dim = inputs[0].dims.nbDims;
    size_t elements_size = 1;
    for(int i = 0; i < n_dim; ++i){
        elements_size *= inputs[1].dims.d[i];
    }

    size_t size = (n_dim * 4 + n_dim * elements_size) * sizeof(int);

    return size;
}

DimsExprs
onnx2trt::GatherElementsPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs,
                                                    IExprBuilder &exprBuilder) {

    assert(outputIndex == 0 && nbInputs == 2);
    return inputs[1];
}

bool
onnx2trt::GatherElementsPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs,
                                                          int32_t nbOutputs) {
    return true;
}

void onnx2trt::GatherElementsPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs,
                                                     const DynamicPluginTensorDesc *out, int32_t nbOutputs) {

    assert(nbInputs == 2 && nbOutputs == 1);
    assert(in[0].desc.dims.nbDims == in[1].desc.dims.nbDims);
}


int32_t onnx2trt::GatherElementsPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
                                                const void *const *inputs, void *const *outputs, void *workspace,
                                                cudaStream_t stream) {

    int n_dim = inputDesc[1].dims.nbDims;
    auto tensor_dims = inputDesc[0].dims.d;
    auto index_dims = inputDesc[1].dims.d;
    gather_elements(inputs, outputs, axis_, n_dim, &tensor_dims[0], &index_dims[0], workspace, stream);
    return 0;
}

onnx2trt::GatherElementsPluginCreator::GatherElementsPluginCreator() = default;

const char* onnx2trt::GatherElementsPluginCreator::getPluginName() const {
    return "GatherElements";
}

const char* onnx2trt::GatherElementsPluginCreator::getPluginVersion() const {
    return "1";
}

const char* onnx2trt::GatherElementsPluginCreator::getPluginNamespace() const {
    return "";
}

IPluginV2* onnx2trt::GatherElementsPluginCreator::deserializePlugin(
        const char *name, const void *serial_data, size_t serial_length) {
    return new GatherElementsPlugin(serial_data, serial_length);
}

void onnx2trt::GatherElementsPluginCreator::setPluginNamespace(const char *N) {}

const PluginFieldCollection* onnx2trt::GatherElementsPluginCreator::getFieldNames() { return nullptr; }

IPluginV2* onnx2trt::GatherElementsPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) {
    return nullptr; }
