#ifndef __INFER_HPP__
#define __INFER_HPP__

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace trt {

#define INFO(...) trt::__log_func(__FILE__, __LINE__, __VA_ARGS__)
void __log_func(const char *file, int line, const char *fmt, ...);

enum class DType : int {
  FLOAT = 0,
  HALF = 1,
  INT8 = 2,
  INT32 = 3,
  BOOL = 4,
  UINT8 = 5
};

class Timer {
public:
  Timer();
  virtual ~Timer();
  void start(void *stream = nullptr);
  float stop(const char *prefix = "Timer", bool print = true);

private:
  void *start_, *stop_;
  void *stream_;
};

class BaseMemory {
public:
  BaseMemory() = default;
  BaseMemory(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes);
  virtual ~BaseMemory();
  virtual void *gpu(size_t bytes);
  virtual void *cpu(size_t bytes);
  void release_gpu();
  void release_cpu();
  void release();
  inline bool owner_gpu() const { return owner_gpu_; }
  inline bool owner_cpu() const { return owner_cpu_; }
  inline size_t cpu_bytes() const { return cpu_bytes_; }
  inline size_t gpu_bytes() const { return gpu_bytes_; }
  virtual inline void *gpu() const { return gpu_; }
  virtual inline void *cpu() const { return cpu_; }
  void reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes);

protected:
  void *cpu_ = nullptr;
  size_t cpu_bytes_ = 0, cpu_capacity_ = 0;
  bool owner_cpu_ = true;

  void *gpu_ = nullptr;
  size_t gpu_bytes_ = 0, gpu_capacity_ = 0;
  bool owner_gpu_ = true;
};

template <typename _DT> class Memory : public BaseMemory {
public:
  Memory() = default;
  Memory(const Memory &other) = delete;
  Memory &operator=(const Memory &other) = delete;
  virtual _DT *gpu(size_t size) override {
    return (_DT*)BaseMemory::gpu(size * sizeof(_DT));
  }
  virtual _DT *cpu(size_t size) override {
    return (_DT*)BaseMemory::cpu(size * sizeof(_DT));
  }

  inline size_t cpu_size() const { return cpu_bytes_ / sizeof(_DT); }
  inline size_t gpu_size() const { return gpu_bytes_ / sizeof(_DT); }

  virtual inline _DT *gpu() const override { return (_DT *)gpu_; }
  virtual inline _DT *cpu() const override { return (_DT *)cpu_; }
};

class Infer {
public:
  virtual bool forward(const std::vector<void *> &bindings,
                       void *stream = nullptr,
                       void *input_consum_event = nullptr) = 0;
  virtual int index(const std::string &name) = 0;
  // run_dims 对应 ExcutionContext-> binding dim
  // ExcutionContext相当于把engine启动起来的效果，想象成windows下面的，启动exe后的进程
  virtual std::vector<int> run_dims(const std::string &name) = 0;
  virtual std::vector<int> run_dims(int ibinding) = 0;
  // static_dims 对应 engine -> binding dim
  // engine对应与你的onnx编译为trtengine那个静态的引擎；engine想象成onnx，没有被启动起来的
  // exe应用程序具体的文件
  // 维度有什么区别：static_dimsengine（onnx的dim）；
  // ！！！最主要的区别，静态的时候可以有不确定的shape，一般不确定的用英文表示；
  // ExcutionContext，rundim属于运行时的状态，必须明确每一个维度；执行的时候还不行，是不行的；
  virtual std::vector<int> static_dims(const std::string &name) = 0;
  virtual std::vector<int> static_dims(int ibinding) = 0;
  virtual int numel(const std::string &name) = 0;
  virtual int numel(int ibinding) = 0;
  virtual int num_bindings() = 0;
  virtual bool is_input(int ibinding) = 0;
  // 你静态的维度不确定，你希望他确定，那么通过set_run_dims来确定每一个维度
  virtual bool set_run_dims(const std::string &name,
                            const std::vector<int> &dims) = 0;
  virtual bool set_run_dims(int ibinding, const std::vector<int> &dims) = 0;
  virtual DType dtype(const std::string &name) = 0;
  virtual DType dtype(int ibinding) = 0;
  virtual bool has_dynamic_dim() = 0;
  virtual void print() = 0;
};

std::shared_ptr<Infer> load(const std::string &file);
std::string format_shape(const std::vector<int> &shape);

} // namespace trt

#endif // __INFER_HPP__
