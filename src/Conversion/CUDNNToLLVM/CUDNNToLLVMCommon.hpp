//===------ CUDNNToLLVM.hpp - Lowering from KRNL+Affine+CUDNN+Std to LLVM -------===//
//
// Copyright 2021 Yonsei CORELAB.
//
// =============================================================================
//
//
//
//===----------------------------------------------------------------------===//

#ifndef CUDNN_TO_LLVM_H
#define CUDNN_TO_LLVM_H

#include "mlir/Support/LLVM.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/Value.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

#include "src/Dialect/CUDNN/CUDNNOps.hpp"
#include "src/Pass/CDPasses.hpp"
#include "src/Support/KrnlSupport.hpp"

// #include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
using namespace mlir;

void generateCUDNNHandle(MLIRContext *context, mlir::ModuleOp &m, Value &cudnnHandle);
void generateCUDAStream(MLIRContext *context, mlir::ModuleOp &m, Value &cudaStreamValue);

namespace mlir {

class MLIRContext;
class LLVMTypeConverter;
class RewritePatternSet;

// Insert the insertValueOp for op output shape information. After make the
// insertValueOps it returns the last insertValueOp for usage of original op.
Value insertAndReturnOutputShapeInfo (
    MLIRContext* context, Location loc, TypeConverter* typeConverter,
    ConversionPatternRewriter &rewriter, Value cudnnOutput, Value llvmOp);

// Insert unrealized conversion cast op to convert memref to llvm struct type.
Value castToLLVMStruct(MLIRContext *context, TypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter, Location &loc, Value v);

void populateCUDNNToLLVMConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle, Value stream);

// ------------------- CUDA ---------------------//
void populateCUDNNMallocToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNMemPoolInitToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNMemOffsetToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNDeallocToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNCastToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNClipToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNMemcpyToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value stream);
void populateCUDNNConcatToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNReciprocalToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNNegativeToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNErfToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNFlattenToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNReshapeToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNSqueezeToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNUnsqueezeToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNTransposeToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNExpandToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNGatherToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNNonZeroToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNPowToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNMatmulNdToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNPReluToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNSoftmaxToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateCUDNNLeakyReluToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);

// ------------------ CUDNN --------------------//
void populateCUDNNConvForwardToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);
void populateCUDNNActivationForwardToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);
void populateCUDNNReduceToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);
void populateCUDNNMaxPoolToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);
void populateCUDNNAveragePoolToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);
void populateCUDNNAddToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);
void populateCUDNNMulToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);
void populateCUDNNSqrtToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);

void populateCUDNNConvBiasActivForwardToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);

// ------------------ cuBLAS --------------------//
void populateCUDNNMatmul2dToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
} // namespace mlir

#endif // CUDNN_TO_LLVM_H
