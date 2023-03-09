import onnx
import onnx.helper as helper
import sys
import os

"""

1,116,8400这个shape
116 -> cx,cy,w,h,80class conf,32mask weight
8400 -> 3个层 grid

infer 需要喂 （1，n，116），所以需要transpose一下

mask 32*160*160
32个160*160的概率图 prob map
还不是二值图
怎么变成instance的图，通过 对每一个box而言的cx,cy,w,h: sigmoid(sum(32weight -> *[32x160x160的图，一个图一个weight],dim=0))
32个weight 乘以32x160x160的图
还需要在32 维度上 累加；累加完了就变成了160*160了；乘完的结果做个sigmoid；就是变成了概率图了；prob map
有了prob map 就能用框crop了，最终crop出来的结果就是mask了
mask = crop([cx,cy,w,h],sigmoid(sum(32weight -> *[32x160x160的图，一个图一个weight],dim=0))
32就是mask dim


"""
def main():

    if len(sys.argv) < 2:
        print("Usage:\n python v8trans.py yolov8n.onnx")
        return 1

    file = sys.argv[1]
    if not os.path.exists(file):
        print(f"Not exist path: {file}")
        return 1

    prefix, suffix = os.path.splitext(file)
    dst = prefix + ".transd" + suffix

    model = onnx.load(file)
    node  = model.graph.node[-1]

    old_output = node.output[0]
    node.output[0] = "pre_transpose"

    for specout in model.graph.output:
        if specout.name == old_output:
            shape0 = specout.type.tensor_type.shape.dim[0]
            shape1 = specout.type.tensor_type.shape.dim[1]
            shape2 = specout.type.tensor_type.shape.dim[2]
            new_out = helper.make_tensor_value_info(
                specout.name,
                specout.type.tensor_type.elem_type,
                [0, 0, 0]
            )
            new_out.type.tensor_type.shape.dim[0].CopyFrom(shape0)
            new_out.type.tensor_type.shape.dim[2].CopyFrom(shape1)
            new_out.type.tensor_type.shape.dim[1].CopyFrom(shape2)
            specout.CopyFrom(new_out)

    model.graph.node.append(
        helper.make_node("Transpose", ["pre_transpose"], [old_output], perm=[0, 2, 1])
    )

    print(f"Model save to {dst}")
    onnx.save(model, dst)
    return 0

if __name__ == "__main__":
    sys.exit(main())