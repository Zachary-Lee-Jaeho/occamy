
#include "src/Conversion/CUDNNToLLVM/CUDNNRuntimeAPI.hpp"
#include "src/Conversion/CUDNNToLLVM/CUDNNToLLVMCommon.hpp"

using namespace mlir;
using namespace core_dnn;

Value mlir::insertAndReturnOutputShapeInfo (
    MLIRContext* context, Location loc, TypeConverter* typeConverter,
    ConversionPatternRewriter &rewriter, Value cudnnOutput, Value llvmOp) {

  auto outputMemRefType = cudnnOutput.getType().cast<mlir::MemRefType>();
  auto elemType = typeConverter->convertType(outputMemRefType.getElementType());
  auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
  auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);
  auto outputMemRefShape = outputMemRefType.getShape();
  auto outputRank = outputMemRefShape.size();

  auto int64Ty = IntegerType::get(context, 64);
  auto int64PtrTy = LLVM::LLVMPointerType::get(int64Ty);
  auto int64ArrayTy = LLVM::LLVMArrayType::get(int64Ty, outputRank);

  auto zero64 = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
      rewriter.getI64IntegerAttr(0));
  auto one64 = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
      rewriter.getI64IntegerAttr(1));

  SmallVector<mlir::Type, 3> outputTys(
      {llvmPtrType, llvmPtrType, int64Ty, int64ArrayTy, int64ArrayTy});
  if (outputRank ==0) {
    // Deal with the non ranked value like single int 64 data
    outputTys.erase(outputTys.end()-1);
    outputTys.erase(outputTys.end()-1);
  }

  auto returnTy = LLVM::LLVMStructType::getLiteral(context, outputTys);

  // Shape, stride info of this memref
  LLVM::ConstantOp* shape = (LLVM::ConstantOp*)malloc(sizeof(LLVM::ConstantOp)*outputRank);
  LLVM::ConstantOp* stride = (LLVM::ConstantOp*)malloc(sizeof(LLVM::ConstantOp)*outputRank);
  int64_t st = 1;
  for (int i = outputRank-1; i >= 0; i--) {
    shape[i] = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(outputMemRefShape[i]));
    stride[i] = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(st));
    st *= outputMemRefShape[i];
  }

  auto undef = rewriter.create<LLVM::UndefOp>(loc, returnTy);
  auto insert0 = rewriter.create<LLVM::InsertValueOp>(loc, returnTy, undef, llvmOp,
      llvm::ArrayRef<int64_t>{0});
  // TODO: fix second element
  auto insert1 = rewriter.create<LLVM::InsertValueOp>(loc, returnTy, insert0, llvmOp,
      llvm::ArrayRef<int64_t>{1});
  auto insert2 = rewriter.create<LLVM::InsertValueOp>(loc, returnTy, insert1, zero64,
      llvm::ArrayRef<int64_t>{2});

  Value insertLast = NULL;
  Value insertTemp = insert2;
  if (outputRank != 0) {
    for (int i=0; i<outputRank; i++) {
      auto insertShape = rewriter.create<LLVM::InsertValueOp>(loc, returnTy, insertTemp, shape[i],
          llvm::ArrayRef<int64_t>{3, i});
      auto insertStride = rewriter.create<LLVM::InsertValueOp>(loc, returnTy, insertShape, stride[i],
          llvm::ArrayRef<int64_t>{4, i});
      insertTemp = insertStride;
      if(i==outputRank-1)
        insertLast = insertStride;
    }
  } else {
    // Deal with the non ranked value like single int 64 data
    insertLast = insert2;
  }

  if (!insertLast)
    assert(0 && "Unreachable point: Last insert value is NULL");
  return insertLast;
}

Value mlir::castToLLVMStruct(MLIRContext *context, TypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter, Location &loc, Value v){
  if (v.getType().dyn_cast_or_null<mlir::MemRefType>()) {
    auto memRefType = v.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(memRefType.getElementType());
    auto llvmelemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto allocPtrType = LLVM::LLVMPointerType::get(llvmelemType);
    auto int64Ty = IntegerType::get(context, 64);
    int64_t rank;
    // XXX: I do not know how to set type when the memRefType is not ranked.
    if (memRefType.hasRank())
      rank = memRefType.getRank();
    else
      rank = 1;
    auto int64ArrayTy = LLVM::LLVMArrayType::get(int64Ty, rank);
    SmallVector<mlir::Type, 3> outputTys(
        {allocPtrType, allocPtrType, int64Ty, int64ArrayTy, int64ArrayTy});
    if (memRefType.getShape().size() == 0) {
      // Deal with the non ranked value like single int 64 data
      outputTys.erase(outputTys.end()-1);
      outputTys.erase(outputTys.end()-1);
    }
    auto convertTy = LLVM::LLVMStructType::getLiteral(context, outputTys);

    return rewriter.create<UnrealizedConversionCastOp>(loc,convertTy,v).getResult(0);
  } else return v;
}

void mlir::populateCUDNNToLLVMConversionPatterns( RewritePatternSet &patterns,
    MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle, Value stream) {
////////////////////////////////// Forward Ops Passes /////////////////////////////////
  // ----------------------------------- CUDA ----------------------------------------//
  mlir::populateCUDNNMallocToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNMemPoolInitToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNMemOffsetToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNDeallocToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNCastToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNClipToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNMemcpyToLLVMConversionPattern(patterns, ctx, typeConverter, stream);
  mlir::populateCUDNNConcatToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNReciprocalToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNNegativeToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNErfToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNFlattenToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNReshapeToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNSqueezeToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNUnsqueezeToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNTransposeToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNExpandToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNGatherToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNNonZeroToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNPowToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNMatmulNdToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNPReluToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNSoftmaxToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateCUDNNLeakyReluToLLVMConversionPattern(patterns, ctx, typeConverter);

  // ------------------------------------ CUDNN --------------------------------------//
  mlir::populateCUDNNConvForwardToLLVMConversionPattern(patterns, ctx, typeConverter, handle);
  mlir::populateCUDNNActivationForwardToLLVMConversionPattern(patterns, ctx, typeConverter, handle);
  mlir::populateCUDNNAddToLLVMConversionPattern(patterns, ctx, typeConverter, handle);
  mlir::populateCUDNNMulToLLVMConversionPattern(patterns, ctx, typeConverter, handle);
  mlir::populateCUDNNSqrtToLLVMConversionPattern(patterns, ctx, typeConverter, handle);
  mlir::populateCUDNNReduceToLLVMConversionPattern(patterns, ctx, typeConverter, handle);
  mlir::populateCUDNNMaxPoolToLLVMConversionPattern(patterns, ctx, typeConverter, handle);
  mlir::populateCUDNNAveragePoolToLLVMConversionPattern(patterns, ctx, typeConverter, handle);

  mlir::populateCUDNNConvBiasActivForwardToLLVMConversionPattern(patterns, ctx, typeConverter, handle);

  // ------------------------------------ cuBLAS --------------------------------------//
  mlir::populateCUDNNMatmul2dToLLVMConversionPattern(patterns, ctx, typeConverter);
}

void generateCUDNNHandle(MLIRContext *context, mlir::ModuleOp &m, Value &cudnnHandle) {
  m.walk([&] (Operation* op){
      if(isa<func::FuncOp>(op)){
        func::FuncOp funcOp = dyn_cast<func::FuncOp>(op);
        auto &parentBlock = funcOp.front();
        OpBuilder builder(&funcOp.front().front());
        const auto &apiRegistry = CUDNNRuntimeAPIRegistry(m, builder, IntegerType::get(context, 32));

        auto handleStructTy =
          LLVM::LLVMStructType::getOpaque("cudnnContext", context);
        auto handlePtrTy =
          LLVM::LLVMPointerType::get(handleStructTy);
        auto handlePtrAddrTy =
          LLVM::LLVMPointerType::get(handlePtrTy);
        auto llvmI32Ty = IntegerType::get(context, 32);

        auto loc = funcOp.front().front().getLoc();

        auto one32 = builder.create<LLVM::ConstantOp>(
            loc, llvmI32Ty, builder.getI32IntegerAttr(1));
        auto handleAlloca = builder.create<LLVM::AllocaOp>(
            loc, handlePtrAddrTy, one32, 0);
        auto handle = CUDNNRuntimeAPI::callApi(builder, loc, apiRegistry,
            CUDNNRuntimeAPI::API::CUDNN_CREATE, {handleAlloca});

        auto handleLoad = builder.create<LLVM::LoadOp>(
            loc, handlePtrTy, handleAlloca);
        auto handleDestroy = CUDNNRuntimeAPI::callApi(builder, loc, apiRegistry,
            CUDNNRuntimeAPI::API::CUDNN_DESTROY, {handleLoad});
        handleLoad.getOperation()->moveBefore(&parentBlock.back());
        handleDestroy.getDefiningOp()->moveBefore(&parentBlock.back());
        cudnnHandle = handleAlloca;
      }
  });
}

void generateCUDAStream(MLIRContext *context, mlir::ModuleOp &m, Value &cudaStreamValue) {
  m.walk([&] (Operation* op){
      if(isa<func::FuncOp>(op)){
        func::FuncOp funcOp = dyn_cast<func::FuncOp>(op);
        auto &parentBlock = funcOp.front();
        OpBuilder builder(&funcOp.front().front());
        const auto &apiRegistry = CUDNNRuntimeAPIRegistry(m, builder, IntegerType::get(context, 32));

        auto streamStructTy =
          LLVM::LLVMStructType::getOpaque("CUstream_st", context);
        auto streamPtrTy =
          LLVM::LLVMPointerType::get(streamStructTy);
        auto streamPtrAddrTy =
          LLVM::LLVMPointerType::get(streamPtrTy);
        auto llvmI32Ty = IntegerType::get(context, 32);

        auto loc = funcOp.front().front().getLoc();

        auto one32 = builder.create<LLVM::ConstantOp>(
            loc, llvmI32Ty, builder.getI32IntegerAttr(1));
        auto streamAlloca = builder.create<LLVM::AllocaOp>(
            loc, streamPtrAddrTy, one32, 0);
        auto stream = CUDNNRuntimeAPI::callApi(builder, loc, apiRegistry,
            CUDNNRuntimeAPI::API::STREAM_CREATE, {streamAlloca});

        auto streamLoad = builder.create<LLVM::LoadOp>(
            loc, streamPtrTy, streamAlloca);
        auto streamDestroy = CUDNNRuntimeAPI::callApi(builder, loc, apiRegistry,
            CUDNNRuntimeAPI::API::STREAM_DESTROY, {streamLoad});
        streamLoad.getOperation()->moveBefore(&parentBlock.back());
        streamDestroy.getDefiningOp()->moveBefore(&parentBlock.back());
        cudaStreamValue = streamAlloca;
      }
  });
}
