#include "model/yolov5.hpp"
#include "common/timer.hpp"
#include "common/image.hpp"
#include "common/position.hpp"

static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) 
{
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) 
    {
        case 0:
            r = v, g = t, b = p;
            break;
        case 1:
            r = q, g = v, b = p;
            break;
        case 2:
            r = p, g = v, b = t;
            break;
        case 3:
            r = p, g = q, b = v;
            break;
        case 4:
            r = t, g = p, b = v;
            break;
        case 5:
            r = v, g = p, b = q;
            break;
        default:
            r = 1, g = 1, b = 1;
            break;
    }
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255),
                        static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) 
{
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

static std::tuple<int, int, int> getFontSize(const std::string& text)
{
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
    return std::make_tuple(textSize.width, textSize.height, baseline);
}

static const char *cocolabels[] = {"head",        "helmet"};

void v5SlicedInfer()
{
    cv::Mat image = cv::imread("inference/persons.jpg");
    auto yolo = yolov5::load("helmetv5.engine");
    if (yolo == nullptr) return;
    auto objs = yolo->forward(tensor::cvimg(image));
    printf("objs size : %d\n", objs.size());
    PositionManager<int> pm(getFontSize);
    for (auto &obj : objs) 
    {
        // std::cout << obj << std::endl;
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                    cv::Scalar(b, g, r), 5);
    }
    for (auto &obj : objs) 
    {
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);
        auto name = cocolabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);

        std::tuple<int, int, int, int> box = std::make_tuple((int)obj.left, (int)obj.top, (int)obj.right, (int)obj.bottom);
        int x, y;
        std::tie(x, y) = pm.selectOptimalPosition(box, image.cols, image.rows, caption);
        std::tuple<int, int, int, int> curPos = pm.getCurrentPosition();
        int left, top, right, bottom;
        std::tie(left, top, right, bottom) = curPos;
        cv::rectangle(image, cv::Point(left, top),
                    cv::Point(right, bottom), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(x, y), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    printf("Save result to result/v5SlicedInfer.jpg, %d objects\n", (int)objs.size());
    cv::imwrite("result/v5SlicedInfer.jpg", image);

}

void v5NoSlicedInfer()
{
    cv::Mat image = cv::imread("inference/persons.jpg");
    auto yolo = yolov5:load("helmetv5.engine");
    if (yolo == nullptr) return;
    auto objs = yolo->forward(tensor::cvimg(image), image.cols, image.rows, 0.0f, 0.0f);
    printf("objs size : %d\n", objs.size());
    PositionManager<int> pm(getFontSize);
    for (auto &obj : objs) 
    {
        // obj.dump();
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                    cv::Scalar(b, g, r), 5);
    }
    for (auto &obj : objs) 
    {
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);        
        auto name = cocolabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);
        std::tuple<int, int, int, int> box = std::make_tuple((int)obj.left, (int)obj.top, (int)obj.right, (int)obj.bottom);
        int x, y;
        std::tie(x, y) = pm.selectOptimalPosition(box, image.cols, image.rows, caption);
        std::tuple<int, int, int, int> curPos = pm.getCurrentPosition();
        int left, top, right, bottom;
        std::tie(left, top, right, bottom) = curPos;
        cv::rectangle(image, cv::Point(left, top),
                    cv::Point(right, bottom), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(x, y), 0, 1, cv::Scalar::all(0), 2, 16);
        
    }
    printf("Save result to result/v5NoSlicedInfer.jpg, %d objects\n", (int)objs.size());
    cv::imwrite("result/v5NoSlicedInfer.jpg", image);

}
