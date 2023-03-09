//
// Created by ros on 3/6/23.
//

#include "yolo.hpp"
#include "cpm.hpp"
#include "infer.hpp"
#include <opencv2/opencv.hpp>


static const char *cocolabels[] = {
        "person",        "bicycle",      "car",
        "motorcycle",    "airplane",     "bus",
        "train",         "truck",        "boat",
        "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench",        "bird",
        "cat",           "dog",          "horse",
        "sheep",         "cow",          "elephant",
        "bear",          "zebra",        "giraffe",
        "backpack",      "umbrella",     "handbag",
        "tie",           "suitcase",     "frisbee",
        "skis",          "snowboard",    "sports ball",
        "kite",          "baseball bat", "baseball glove",
        "skateboard",    "surfboard",    "tennis racket",
        "bottle",        "wine glass",   "cup",
        "fork",          "knife",        "spoon",
        "bowl",          "banana",       "apple",
        "sandwich",      "orange",       "broccoli",
        "carrot",        "hot dog",      "pizza",
        "donut",         "cake",         "chair",
        "couch",         "potted plant", "bed",
        "dining table",  "toilet",       "tv",
        "laptop",        "mouse",        "remote",
        "keyboard",      "cell phone",   "microwave",
        "oven",          "toaster",      "sink",
        "refrigerator",  "book",         "clock",
        "vase",          "scissors",     "teddy bear",
        "hair drier",    "toothbrush"};

int main(){

    // auto model = trt::load("yolov5s.engine");
    // model->print();

    auto image = cv::imread("inference/gril.jpg");
    // auto model = yolo::load("yolov5s.engine",yolo::Type::V5);  // 底层还是调用的trt的load，只是输入输出预处理给封装了
    // auto objs = model->forward(yolo::Image(image.data,image.cols,image.rows));  // 其实还有个stream的参数想，现在先不给了，同步的forward，拿到的就是框的数量

    // 下面就是异步推理
    cpm::Instance<yolo::BoxArray,yolo::Image,yolo::Infer> cpmi;
    bool ok = cpmi.start(
            []{return yolo::load("yolov5s.engine",yolo::Type::V5);}
            );
    if (!ok){
        return 1;
    }
    trt::Timer timer;
    // warmup 下
    for (int i = 0; i < 10; ++i){
        timer.start();
        // 直接改成commits就是多batch
        auto fut = cpmi.commit(yolo::Image(image.data,image.cols,image.rows));
        auto objs = fut.get();
        timer.stop("Commit");  // 耗时自动打印出来
    }
    timer.start();
    // 直接改成commits就是多batch
    auto fut = cpmi.commit(yolo::Image(image.data,image.cols,image.rows));
    auto objs = fut.get();
    timer.stop("Commit");  // 耗时自动打印出来
    printf("objs is %d\n",objs.size());
    for (auto &obj : objs) {
        uint8_t b, g, r;
        std::tie(b, g, r) = yolo::random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top),
                      cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

        auto name = cocolabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);
        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                      cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r),
                      -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1,
                    cv::Scalar::all(0), 2, 16);
    }
    printf("Save result to Result.jpg, %d objects\n", objs.size());
    // cv::imwrite(cv::format("Result%d.jpg", ib), image);
    cv::imwrite("Result.jpg", image);

    return 0;

}



