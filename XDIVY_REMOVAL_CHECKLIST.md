# Xdivy 算子移除清单

## ✅ 已完成的删除

### 1. Python API 层
- ✅ `mindspore/python/mindspore/ops/operations/math_ops.py` - 删除 Xdivy 类定义
- ✅ `mindspore/python/mindspore/ops/operations/__init__.py` - 删除 Xdivy 导入和导出
- ✅ `mindspore/python/mindspore/ops/function/math_func.py` - 删除 xdivy 函数和 xdivy_ 实例
- ✅ `mindspore/python/mindspore/ops/function/__init__.py` - 删除 xdivy 导出
- ✅ `mindspore/python/mindspore/ops/functional.py` - 删除 xdivy 注册
- ✅ `mindspore/python/mindspore/common/tensor.py` - 删除 Tensor.xdivy 方法
- ✅ `mindspore/python/mindspore/_extends/parse/standard_method.py` - 删除 xdivy 函数
- ✅ `mindspore/python/mindspore/ops/_vmap/vmap_math_ops.py` - 删除 Xdivy vmap 注册
- ✅ `mindspore/python/mindspore/ops/_op_impl/aicpu/__init__.py` - 删除 xdivy 导入
- ✅ `mindspore/python/mindspore/ops/_op_impl/aicpu/xdivy.py` - 完全删除文件

### 2. Kernel 实现
- ✅ `mindspore/ops/kernel/cpu/native/xdivy_cpu_kernel.h` - 完全删除
- ✅ `mindspore/ops/kernel/cpu/native/xdivy_cpu_kernel.cc` - 完全删除
- ✅ `mindspore/ops/kernel/ascend/aicpu/aicpu_ops/cpu_kernel/ms_kernel/xdivy.h` - 完全删除
- ✅ `mindspore/ops/kernel/ascend/aicpu/aicpu_ops/cpu_kernel/ms_kernel/xdivy.cc` - 完全删除

### 3. 算子定义
- ✅ `mindspore/ops/infer/xdivy.h` - 完全删除
- ✅ `mindspore/ops/infer/xdivy.cc` - 完全删除

### 4. 测试文件
- ✅ `tests/st/ops/gpu/test_xdivy_op.py` - 完全删除
- ✅ `tests/st/ops/cpu/test_xdivy_op.py` - 完全删除
- ✅ `tests/st/ops/dynamic_shape/grad/test_xdivy.py` - 完全删除
- ✅ `tests/st/ops/gpu/test_xdivy_xlogy_op.py` - 完全删除
- ✅ `tests/ut/python/ops/test_ops.py` - 删除 Xdivy 测试用例
- ✅ `tests/ut/python/parallel/test_arithmetic.py` - 删除 test_matmul_xdivy_broadcast 函数

### 5. 文档文件
- ✅ `docs/api/api_python/ops/mindspore.ops.func_xdivy.rst` - 完全删除
- ✅ `docs/api/api_python/ops/mindspore.ops.Xdivy.rst` - 完全删除
- ✅ `docs/api/api_python/mindspore/Tensor/mindspore.Tensor.xdivy.rst` - 完全删除

---

## ⚠️ 需要手动处理的 C++ 文件

以下文件包含 xdivy 相关代码，需要手动编辑删除相关部分：

### GPU Kernel 实现
1. **`mindspore/ops/kernel/gpu/cuda/math/binary_ops_gpu_kernel.h`**
   - 删除 Xdivy 相关的模板特化和声明

2. **`mindspore/ops/kernel/gpu/cuda/math/binary_ops_gpu_kernel.cc`**
   - 删除 Xdivy kernel 注册
   - 删除 MS_REG_GPU_KERNEL_TWO 中的 Xdivy 注册

3. **`mindspore/ops/kernel/gpu/cuda_impl/cuda_ops/binary_types.cuh`**
   - 删除 BinaryOpType::kXdivy 枚举值

4. **`mindspore/ops/kernel/gpu/cuda_impl/cuda_ops/binary_divs_func.cu`**
   - 删除 Xdivy CUDA kernel 实现

### 算子定义
5. **`mindspore/ops/op_def/math_ops.h`**
   - 删除 Xdivy 算子定义声明

6. **`mindspore/ops/op_def/math_op_name.h`**
   - 删除 kNameXdivy 常量定义

### Ascend 适配
7. **`mindspore/ccsrc/plugin/ascend/res_manager/op_adapter/op_adapter_map.h`**
   - 删除 Xdivy 算子映射

8. **`mindspore/ccsrc/plugin/ascend/res_manager/op_adapter/op_declare/elewise_calculation_ops_declare.h`**
   - 删除 Xdivy 算子声明

9. **`mindspore/ccsrc/plugin/ascend/res_manager/op_adapter/op_declare/elewise_calculation_ops_declare.cc`**
   - 删除 Xdivy 算子实现

10. **`mindspore/ops/kernel/ascend/aicpu/aicpu_ops/customize/op_info_cfg/cust_aicpu_kernel.ini`**
    - 删除 Xdivy 配置项

11. **`mindspore/ops/kernel/ascend/aicpu/aicpu_ops/customize/utils/aicpu_parser_ini.py`**
    - 可能需要删除 xdivy 相关的解析逻辑（如果有）

### 并行和优化
12. **`mindspore/ccsrc/frontend/parallel/ops_info/ops_utils.h`**
    - 删除 Xdivy 相关的并行策略声明

13. **`mindspore/ccsrc/frontend/parallel/ops_info/arithmetic_info.h`**
    - 删除 XdivyInfo 类声明

14. **`mindspore/ccsrc/frontend/parallel/ops_info/arithmetic_info.cc`**
    - 删除 XdivyInfo 类实现

15. **`mindspore/ccsrc/frontend/parallel/step_parallel_utils.cc`**
    - 删除 Xdivy 相关的并行处理逻辑

16. **`mindspore/ccsrc/frontend/parallel/auto_parallel/operator_costmodel.h`**
    - 删除 Xdivy 代价模型（如果有）

### 图编译和优化
17. **`mindspore/ccsrc/frontend/expander/grad/grad_math_ops.cc`**
    - 删除 Xdivy 梯度展开实现

18. **`mindspore/ccsrc/include/utils/expander/emitter.h`**
    - 删除 Xdivy 相关的 emitter 声明（如果有）

19. **`mindspore/ccsrc/frontend/jit/ps/resource.cc`**
    - 删除 Xdivy 相关的资源管理代码（如果有）

### 配置文件
20. **`config/op_info.config`**
    - 删除 Xdivy 算子配置项

### 文档（英文）
21. **`docs/api/api_python_en/mindspore/mindspore.Tensor.rst`**
    - 删除 xdivy 方法引用

22. **`docs/api/api_python_en/mindspore.ops.rst`**
    - 删除 xdivy 函数引用

23. **`docs/api/api_python_en/mindspore.ops.primitive.rst`**
    - 删除 Xdivy 类引用

24. **`docs/api/api_python/mindspore/mindspore.Tensor.rst`**
    - 删除 xdivy 方法引用

25. **`docs/api/api_python/mindspore.ops.rst`**
    - 删除 xdivy 函数引用

26. **`docs/api/api_python/mindspore.ops.primitive.rst`**
    - 删除 Xdivy 类引用

### Release Notes
27. **`RELEASE_CN.md`**
    - 如果有 xdivy 相关的发布说明，需要删除或标记为已废弃

28. **`RELEASE.md`**
    - 如果有 xdivy 相关的发布说明，需要删除或标记为已废弃

---

## 📝 删除步骤建议

### 对于 C++ 文件：
1. 搜索 `xdivy`、`Xdivy`、`XDIVY` 关键字
2. 删除相关的：
   - 类定义和声明
   - 函数实现
   - 宏注册（如 MS_REG_GPU_KERNEL）
   - 枚举值
   - 配置项
3. 删除相关的 include 语句（如果该头文件只被 xdivy 使用）

### 对于配置文件：
1. 删除包含 "xdivy" 或 "Xdivy" 的整行配置

### 对于文档文件：
1. 删除 xdivy 相关的章节、示例和引用
2. 更新目录（如果有）

---

## 🔍 验证步骤

完成所有删除后，执行以下验证：

```bash
# 1. 搜索残留引用
grep -r "xdivy" mindspore/ --exclude-dir=build
grep -r "Xdivy" mindspore/ --exclude-dir=build
grep -r "XDIVY" mindspore/ --exclude-dir=build

# 2. 编译测试
cd mindspore
bash build.sh -e cpu -j8

# 3. 运行相关测试
pytest tests/st/ops/cpu/ -v
pytest tests/st/ops/gpu/ -v

# 4. 检查文档构建
cd docs
make html
```

---

## ⚠️ 注意事项

1. **备份**：在删除前建议创建 git 分支或备份
2. **依赖检查**：确保没有其他算子依赖 xdivy
3. **API 兼容性**：这是一个破坏性变更，需要在 Release Notes 中说明
4. **文档更新**：需要更新迁移指南，告知用户使用替代方案（如 `x / y` 或 `ops.div`）

---

## 📊 统计

- **已删除文件**: 13 个
- **已修改 Python 文件**: 10 个
- **待处理 C++ 文件**: ~20 个
- **待处理配置/文档**: ~8 个

---

生成时间: 2025-01-XX

