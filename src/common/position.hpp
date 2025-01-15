#ifndef POSITION_HPP__
#define POSITION_HPP__

#include <vector>
#include <string>
#include <tuple>


template <typename T>
float computeIoU(const std::tuple<T, T, T, T>& box1, const std::tuple<T, T, T, T>& box2)
{
    T l1, t1, r1, b1;
    T l2, t2, r2, b2;
    std::tie(l1, t1, r1, b1) = box1;
    std::tie(l2, t2, r2, b2) = box2;

    T areaA = (r1 - l1) * (b1 - t1);
    T areaB = (r2 - l2) * (b2 - t2);
    T intersectionWidth = std::max(T(0), std::min(r1, r2) - std::max(l1, l2));
    T intersectionHeight = std::max(T(0), std::min(b1, b2) - std::max(t1, t2));
    T intersectionArea = intersectionWidth * intersectionHeight;

    if (areaA == 0 || areaB == 0 || intersectionArea == 0) {
        return 0;
    }
    return static_cast<float>(intersectionArea) / (areaA + areaB - intersectionArea);
}


template <typename T>
float computeOverlap(const std::tuple<T, T, T, T>& box1, const std::tuple<T, T, T, T>& box2)
{
    T l1, t1, r1, b1;
    T l2, t2, r2, b2;
    std::tie(l1, t1, r1, b1) = box1;
    std::tie(l2, t2, r2, b2) = box2;

    T areaA = (r1 - l1) * (b1 - t1);
    T areaB = (r2 - l2) * (b2 - t2);
    T intersectionWidth = std::max(T(0), std::min(r1, r2) - std::max(l1, l2));
    T intersectionHeight = std::max(T(0), std::min(b1, b2) - std::max(t1, t2));
    T intersectionArea = intersectionWidth * intersectionHeight;

    if (areaA == 0 || areaB == 0 || intersectionArea == 0) {
        return 0;
    }
    return static_cast<float>(intersectionArea) / std::min(areaA, areaB);
}

template <typename T>
class PositionManager{
private:
    std::vector<std::tuple<T, T, T, T>> markedPositions;

public:
    std::tuple<T, T> selectOptimalPosition(const std::tuple<T, T, T, T>& box,
        int canvasWidth, int canvasHeight, int textWidth, int textHeight, int baseline)
    {
        std::vector<std::tuple<T, T, T, T>> candidatePositions = 
            findCandidatePositions(box,canvasWidth, canvasHeight, textWidth, textHeight, baseline);

        
        float minIoU = 1.f;
        std::tuple<T, T, T, T> candidatePosition = candidatePositions[0];
        for (const auto& cposition : candidatePositions)
        {
            float maxIoU = 0.f;
            for (const auto& mposition : markedPositions)
            {
                float IoU = computeIoU(cposition, mposition);
                maxIoU = std::max(IoU, maxIoU);
            }
            if (maxIoU == 0.f)
            {
                candidatePosition = cposition;
                break;
            }
            else if (maxIoU < minIoU)
            {
                minIoU  = maxIoU;
                candidatePosition = cposition;
            }
        }
        markedPositions.push_back(candidatePosition);

        T left, top, right, bottom;
        std::tie(left, top, right, bottom) = candidatePosition;
        std::tuple<T, T> result = std::make_tuple(left, top + textHeight);
        return result;
    }

    void clearMarkedPositions()
    {
        markedPositions.clear();
    }

    // 获取候选区域 并附带画图位置起始点
    std::vector<std::tuple<T, T, T, T>> findCandidatePositions(
        const std::tuple<T, T, T, T>& box, 
        int canvasWidth, int canvasHeight, int textWidth, int textHeight, int baseline)
    {
        std::vector<std::tuple<T, T, T, T>> candidatePositions;
        T left, top, right, bottom;
        std::tie(left, top, right, bottom) = box;
        left   = std::max(0, left);
        top    = std::max(0, top);
        right  = std::min(canvasWidth, right);
        bottom = std::min(canvasHeight, bottom);
        std::tuple<T, T, T, T> all = std::make_tuple(0, 0, canvasWidth, canvasHeight);

        // 一个框有6个可以画框的区域，判断那些区域没超过画面
        std::vector<std::tuple<T, T, T, T>> positions = 
        {
            std::make_tuple(left, top - textHeight - baseline, left + textWidth, top),
            std::make_tuple(right, top, right + textWidth, top + textHeight + baseline),
            std::make_tuple(left - textWidth, top, left, top + textHeight + baseline),
            std::make_tuple(left, bottom, left + textWidth, bottom + textHeight + baseline),
            std::make_tuple(right - textWidth, top - textHeight - baseline, right, top),
            std::make_tuple(right - textWidth, bottom, right, bottom + textHeight + baseline),
            std::make_tuple(left, top, left + textWidth, top + textHeight + baseline),
            std::make_tuple(right - textWidth, top, right, top + textHeight + baseline),
            std::make_tuple(right - textWidth, bottom - textHeight - baseline, right, bottom),
            std::make_tuple(left, bottom - textHeight - baseline, left + textWidth, bottom)
        };

        for (const auto& position : positions)
        {
            if (computeOverlap(all, position) == 1.0f)
            {
                candidatePositions.push_back(position);
            }
        }
        if (candidatePositions.size() == 0)
        {
            candidatePositions.push_back(std::make_tuple(left, top, left + textWidth, top + textHeight + baseline));
        }
        return candidatePositions;
    }
};

#endif