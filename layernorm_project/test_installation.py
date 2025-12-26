"""
测试安装脚本
检查所有依赖是否正确安装
"""

import sys

def check_imports():
    """检查必要的库是否可以导入"""
    print("="*60)
    print("检查Python包安装情况")
    print("="*60)
    
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
    }
    
    all_installed = True
    issues = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:20s} 已安装")
        except ImportError as e:
            print(f"✗ {name:20s} 未安装")
            all_installed = False
            issues.append((name, "未安装", str(e)))
        except Exception as e:
            # 捕获其他错误，如版本兼容性问题
            error_msg = str(e)
            if 'pyarrow' in error_msg and 'PyExtensionType' in error_msg:
                print(f"⚠ {name:20s} 已安装但存在版本兼容问题 (pyarrow)")
                issues.append((name, "版本兼容", "pyarrow版本不兼容"))
            else:
                print(f"⚠ {name:20s} 已安装但存在问题: {error_msg[:50]}")
                issues.append((name, "其他错误", error_msg))
    
    # 打印问题和解决方案
    if issues:
        print("\n" + "="*60)
        print("发现的问题和解决方案:")
        print("="*60)
        for name, issue_type, detail in issues:
            if name == "Datasets" and "pyarrow" in detail:
                print(f"\n{name}: pyarrow版本兼容性问题")
                print("  解决方案1 (推荐): 升级pyarrow")
                print("    pip install --upgrade pyarrow")
                print("  解决方案2: 降级pyarrow到兼容版本")
                print("    pip install pyarrow==12.0.1")
                print("  解决方案3: 重新安装datasets")
                print("    pip install --upgrade datasets")
                print("  注意: datasets不是核心依赖，可以跳过IMDB数据集测试")
            elif issue_type == "未安装":
                print(f"\n{name}: 未安装")
                print(f"  解决方案: pip install {name.lower()}")
    
    return all_installed or len([i for i in issues if i[0] != "Datasets"]) == 0  # datasets是可选的


def check_cuda():
    """检查CUDA是否可用"""
    print("\n" + "="*60)
    print("检查CUDA环境")
    print("="*60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA可用")
            print(f"  - CUDA版本: {torch.version.cuda}")
            print(f"  - GPU数量: {torch.cuda.device_count()}")
            print(f"  - GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"  - 显存大小: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # 测试简单的CUDA操作
            x = torch.randn(100, 100).cuda()
            y = x @ x.t()
            torch.cuda.synchronize()
            print(f"✓ CUDA操作测试通过")
            
            return True
        else:
            print("✗ CUDA不可用")
            print("  请检查:")
            print("  1. 是否安装了NVIDIA GPU驱动")
            print("  2. 是否安装了支持CUDA的PyTorch版本")
            print("  3. 环境变量是否正确设置")
            return False
            
    except Exception as e:
        print(f"✗ CUDA检查失败: {e}")
        return False


def check_pytorch_version():
    """检查PyTorch版本"""
    print("\n" + "="*60)
    print("检查PyTorch版本")
    print("="*60)
    
    try:
        import torch
        
        version = torch.__version__
        print(f"PyTorch版本: {version}")
        
        # 检查是否从源码编译
        debug_info = torch.__config__.show()
        
        if "CXX_FLAGS" in debug_info or "BUILD_TYPE" in debug_info:
            print("✓ PyTorch可能是从源码编译的")
        else:
            print("⚠ PyTorch可能不是从源码编译的")
            print("  建议从源码编译以便深入研究")
        
        return True
        
    except Exception as e:
        print(f"✗ 版本检查失败: {e}")
        return False


def check_custom_extension():
    """检查自定义CUDA扩展是否编译"""
    print("\n" + "="*60)
    print("检查自定义CUDA扩展")
    print("="*60)
    
    try:
        import custom_layernorm_cuda
        print("✓ 自定义CUDA扩展已编译")
        
        # 测试基本功能
        import torch
        if torch.cuda.is_available():
            x = torch.randn(2, 4, 768).cuda()
            gamma = torch.ones(768).cuda()
            beta = torch.zeros(768).cuda()
            
            try:
                output = custom_layernorm_cuda.forward_basic(x, gamma, beta, 1e-5)
                print("✓ CUDA扩展功能测试通过")
                return True
            except Exception as e:
                print(f"✗ CUDA扩展功能测试失败: {e}")
                return False
        else:
            print("⚠ 无法测试CUDA扩展（CUDA不可用）")
            return True
            
    except ImportError:
        print("✗ 自定义CUDA扩展未编译")
        print("  请运行: python setup.py install")
        return False


def check_model_download():
    """检查是否可以下载模型"""
    print("\n" + "="*60)
    print("检查模型下载")
    print("="*60)
    
    try:
        from transformers import BertTokenizer
        
        print("尝试加载bert-base-uncased tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        print("✓ 可以正常下载和加载模型")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型下载失败: {e}")
        print("  可能的原因:")
        print("  1. 网络连接问题")
        print("  2. Hugging Face访问受限")
        print("  建议:")
        print("  1. 使用代理或镜像")
        print("  2. 手动下载模型文件")
        return False


def print_summary(results):
    """打印总结"""
    print("\n" + "="*60)
    print("安装检查总结")
    print("="*60)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{check:30s} {status}")
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✓ 所有检查都通过!")
        print("  可以开始运行实验了")
        print("\n建议的运行顺序:")
        print("  1. python layernorm_research.py")
        print("  2. python bert_inference_benchmark.py")
        print("  3. python setup.py install  (如果还没编译)")
        print("  4. python performance_comparison.py")
    else:
        print("✗ 部分检查未通过")
        print("  请根据上面的提示解决问题")
    
    print("="*60)


def main():
    """主函数"""
    print("\n" + "#"*60)
    print("# LayerNorm项目安装检查")
    print("#"*60 + "\n")
    
    results = {
        'Python包': check_imports(),
        'CUDA环境': check_cuda(),
        'PyTorch版本': check_pytorch_version(),
        '自定义扩展': check_custom_extension(),
        '模型下载': check_model_download(),
    }
    
    print_summary(results)


if __name__ == "__main__":
    main()

