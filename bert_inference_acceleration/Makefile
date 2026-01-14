# Makefile for BERT Inference Acceleration Project

.PHONY: all install clean test benchmark inference help

# 默认目标
all: install test

# 安装依赖和编译算子
install:
	@echo "安装Python依赖..."
	pip install -r requirements.txt
	@echo "编译自定义CUDA算子..."
	cd custom_ops && python setup.py install
	@echo "安装完成！"

# 清理编译产物
clean:
	@echo "清理编译产物..."
	cd custom_ops && rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.so" -delete 2>/dev/null || true
	@echo "清理完成！"

# 运行正确性测试
test:
	@echo "运行正确性测试..."
	python tests/test_correctness.py

# 运行性能测试
benchmark:
	@echo "运行性能测试..."
	python benchmarks/benchmark.py --num_iters 100

# 运行IMDB推理
inference:
	@echo "运行IMDB推理测试..."
	python inference.py --batch_size 32 --num_samples 1000

# 运行示例
example:
	@echo "运行使用示例..."
	python examples/usage_example.py

# 快速测试（较少迭代）
quick-test:
	@echo "运行快速测试..."
	python benchmarks/benchmark.py --num_iters 20

# 完整流程
full:
	@echo "执行完整测试流程..."
	$(MAKE) install
	$(MAKE) test
	$(MAKE) benchmark
	$(MAKE) inference

# 帮助信息
help:
	@echo "BERT推理加速项目 - Makefile命令"
	@echo ""
	@echo "可用命令:"
	@echo "  make install       - 安装依赖并编译CUDA算子"
	@echo "  make clean         - 清理编译产物"
	@echo "  make test          - 运行正确性测试"
	@echo "  make benchmark     - 运行性能测试"
	@echo "  make inference     - 运行IMDB推理测试"
	@echo "  make example       - 运行使用示例"
	@echo "  make quick-test    - 快速性能测试"
	@echo "  make full          - 执行完整测试流程"
	@echo "  make help          - 显示此帮助信息"
	@echo ""
	@echo "默认: make (等同于 make install test)"




