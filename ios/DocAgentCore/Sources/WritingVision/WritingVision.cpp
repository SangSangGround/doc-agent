// Mobile implementation of hand-tracking/detection/document_tracker_v7{,_3,_5}.
// The reader's edge detector is deliberately independent of this reference matcher.
#include "WritingVision.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace {
using Points = std::vector<cv::Point2f>;

struct Features {
    cv::Size size;
    std::vector<cv::KeyPoint> keys;
    cv::Mat descriptors;
};

bool validImage(const uint8_t* bytes, int w, int h, size_t stride) {
    return bytes && w >= 48 && h >= 48 && w <= 12000 && h <= 12000 && stride >= static_cast<size_t>(w);
}

cv::Mat imageView(const uint8_t* gray, int w, int h, size_t stride) {
    return cv::Mat(h, w, CV_8UC1, const_cast<uint8_t*>(gray), stride);
}

Points pixels(const Points& unit, cv::Size size) {
    Points result;
    for (const auto& p : unit) result.emplace_back(p.x * size.width, p.y * size.height);
    return result;
}

double median(std::vector<float>& v) {
    auto at = v.begin() + v.size()/2;
    std::nth_element(v.begin(), at, v.end());
    return *at;
}

double medianError(const Points& src, const Points& dst, const cv::Mat& homography) {
    Points projected;
    cv::perspectiveTransform(src, projected, homography);
    std::vector<float> errors;
    for (size_t i = 0; i < dst.size(); ++i) errors.push_back(cv::norm(projected[i] - dst[i]));
    return median(errors);
}

cv::Mat scale(cv::Size size) {
    return (cv::Mat_<double>(3,3) << size.width,0,0, 0,size.height,0, 0,0,1);
}

cv::Mat thumbnail(const cv::Mat& gray) {
    cv::Mat small;
    double factor = std::min(1., 520./std::max(gray.rows, gray.cols));
    cv::resize(gray, small, cv::Size(), factor, factor, cv::INTER_AREA);
    return small;
}

cv::Mat ink(const cv::Mat& image) {
    cv::Mat background, foreground;
    cv::GaussianBlur(image, background, cv::Size(), 3.5);
    cv::subtract(background, image, foreground);
    cv::GaussianBlur(foreground, foreground, cv::Size(3,3), .7);
    return foreground;
}
}

struct DAWritingTracker {
    cv::Ptr<cv::ORB> orb = cv::ORB::create(3500, 1.2f, 8, 20, 0, 2, cv::ORB::HARRIS_SCORE, 31, 11);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2, cv::Size(8,8));
    cv::BFMatcher matcher{cv::NORM_HAMMING, false};
    Points page;
    Features reference, detail;
    cv::Mat referenceInk;

    // Repeated table rules and shared headers can yield a plausible geometric
    // transform on a different form. Require agreement across visible printed
    // regions too. Unseen portions of a close-up and plain white cells do not vote.
    bool samePrint(const cv::Mat& gray, const cv::Mat& normalized) const {
        cv::Mat live = thumbnail(gray), aligned, visible;
        cv::Mat transform = scale(live.size()) * normalized * scale(referenceInk.size()).inv();
        cv::warpPerspective(live, aligned, transform, referenceInk.size(),
                            cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, 255);
        cv::warpPerspective(cv::Mat(live.size(), CV_8U, cv::Scalar(255)), visible,
                            transform, referenceInk.size(), cv::INTER_NEAREST | cv::WARP_INVERSE_MAP);
        cv::Mat actual = ink(aligned);
        auto paper = pixels(page,referenceInk.size());
        int compared = 0, matched = 0;
        for (int row = 0; row < 10; ++row) {
            for (int col = 0; col < 8; ++col) {
                int w = referenceInk.cols / 8, h = referenceInk.rows / 10;
                cv::Rect cell(col*w, row*h, w, h);
                if (cv::pointPolygonTest(paper, cv::Point2f(cell.x+w*.5, cell.y+h*.5), false) < 0 ||
                    cv::countNonZero(visible(cell)) < .94*cell.area()) continue;
                auto ref = referenceInk(cell);
                if (cv::norm(ref,cv::NORM_L2) < 6.0*std::sqrt(cell.area())) continue;
                cv::Mat correlation;
                cv::matchTemplate(actual(cell), ref, correlation, cv::TM_CCORR_NORMED);
                ++compared;
                if (correlation.at<float>(0,0) >= .65f) ++matched;
            }
        }
        return compared >= 4 && matched >= .58*compared;
    }

    Features extract(const cv::Mat& gray, int longest, bool isReference) {
        double factor = std::min(1., double(longest)/std::max(gray.cols, gray.rows));
        cv::Mat small, equalized, mask;
        cv::resize(gray, small, cv::Size(), factor, factor, cv::INTER_AREA);
        clahe->apply(small, equalized);
        if (isReference) {
            std::vector<cv::Point> polygon;
            for (auto p : pixels(page, small.size())) polygon.emplace_back(cvRound(p.x),cvRound(p.y));
            mask = cv::Mat::zeros(small.size(), CV_8U);
            cv::fillConvexPoly(mask, polygon, cv::Scalar(255));
            cv::erode(mask, mask, cv::Mat::ones(11, 11, CV_8U));
        }
        Features f;
        f.size = small.size();
        orb->detectAndCompute(equalized, mask, f.keys, f.descriptors);
        return f;
    }

    bool match(const Features& ref, const Features& live, bool allowWhole, cv::Mat& normalized) {
        if (ref.descriptors.empty() || live.keys.size() < 35) return false;
        std::vector<std::vector<cv::DMatch>> pairs;
        matcher.knnMatch(ref.descriptors, live.descriptors, pairs, 2);
        return (allowWhole && verify(ref, live, pairs, false, normalized)) ||
               verify(ref, live, pairs, true, normalized);
    }

    bool verify(const Features& ref, const Features& live,
                const std::vector<std::vector<cv::DMatch>>& pairs, bool partial, cv::Mat& normalized) {
        // Disallow multiple votes for one identical live keypoint (repeated glyphs/checkboxes).
        std::vector<cv::DMatch> best(live.keys.size());
        for (const auto& pair : pairs) {
            if (pair.size() < 2 || pair[0].distance >= (partial ? .72 : .76)*pair[1].distance) continue;
            auto& previous = best[pair[0].trainIdx];
            if (previous.queryIdx < 0 || previous.distance > pair[0].distance) previous = pair[0];
        }
        Points source, dest;
        for (const auto& m : best) {
            if (m.queryIdx < 0) continue;
            source.push_back(ref.keys[m.queryIdx].pt);
            dest.push_back(live.keys[m.trainIdx].pt);
        }
        if (source.size() < (partial ? 28u : 18u)) return false;
        cv::Mat mask;
        auto h = cv::findHomography(source, dest, cv::RANSAC, partial ? 3.5 : 5., mask, 3000, .995);
        if (h.empty() || !cv::checkRange(h)) return false;
        Points inRef, inLive;
        const auto* selected = mask.ptr<uint8_t>();
        for (size_t i = 0; i < source.size(); ++i) {
            if (!selected[i]) continue;
            inRef.push_back(source[i]);
            inLive.push_back(dest[i]);
        }
        if (inRef.size() < (partial ? 24u : 18u) || inRef.size() < (partial ? .53 : .39)*source.size()) return false;
        auto pagePixels = pixels(page, ref.size);
        auto sheet = cv::boundingRect(pagePixels);
        auto coverage = cv::boundingRect(inRef);
        if (coverage.width < (partial ? .12 : .27)*sheet.width || coverage.height < (partial ? .11 : .27)*sheet.height) return false;
        Points hull;
        cv::convexHull(inLive, hull);
        auto liveCoverage = cv::boundingRect(hull);
        if (liveCoverage.width < .22*live.size.width || liveCoverage.height < .17*live.size.height ||
            cv::contourArea(hull) < .023*live.size.area()) return false;
        if (medianError(inRef, inLive, h) > (partial ? 2.7 : 3.7) || medianError(inLive, inRef, h.inv()) > 4.) return false;
        Points q;
        cv::perspectiveTransform(pagePixels, q, h);
        if (!cv::checkRange(cv::Mat(q)) || !cv::isContourConvex(q) || cv::contourArea(q, true) < .06*live.size.area()) return false;
        if (!partial) for (const auto& p : q)
            if (p.x < -.03*live.size.width || p.x > 1.03*live.size.width || p.y < -.03*live.size.height || p.y > 1.03*live.size.height) return false;
        normalized = scale(live.size).inv() * h * scale(ref.size);
        return cv::checkRange(normalized) && std::abs(cv::determinant(normalized)) > 1e-9;
    }
};

DAWritingTracker* DAWritingCreate(const uint8_t* gray,int w,int h,size_t stride,const double corners[8]) {
    if (!validImage(gray,w,h,stride) || !corners) return nullptr;
    try {
        std::unique_ptr<DAWritingTracker> tracker(new DAWritingTracker());
        for (int i=0; i<4; ++i) tracker->page.emplace_back(corners[i*2],corners[i*2+1]);
        if (!cv::isContourConvex(tracker->page) || cv::contourArea(tracker->page, true) < .035) return nullptr;
        cv::Mat src = imageView(gray,w,h,stride);
        tracker->referenceInk = ink(thumbnail(src));
        tracker->reference = tracker->extract(src,1080,true);
        if (std::max(w,h) > 1080) tracker->detail = tracker->extract(src,1600,true);
        if (tracker->reference.keys.size() < 35 && tracker->detail.keys.size() < 35) return nullptr;
        return tracker.release();
    } catch (const std::exception&) { return nullptr; }
}

void DAWritingDestroy(DAWritingTracker *tracker) { delete tracker; }

bool DAWritingRegister(DAWritingTracker *tracker,const uint8_t *gray,int w,int h,size_t stride,double output[9]) {
    if (!tracker || !output || !validImage(gray,w,h,stride)) return false;
    try {
        auto src = imageView(gray,w,h,stride);
        auto live = tracker->extract(src,1080,false);
        cv::Mat normalized;
        if (!tracker->match(tracker->reference,live,true,normalized)) {
            if (tracker->detail.descriptors.empty()) return false;
            auto larger = std::max(w,h)>1080 ? tracker->extract(src,1600,false) : live;
            if (!tracker->match(tracker->detail,larger,false,normalized)) return false;
        }
        if (!tracker->samePrint(src,normalized)) return false;
        std::copy_n(normalized.ptr<double>(),9,output);
        return true;
    } catch (const std::exception&) { return false; }
}

bool DAWritingPaperCoordinates(const double corners[8], double output[9]) {
    if (!corners || !output) return false;
    try {
        cv::Point2f source[4], paper[] = {{0,0}, {1,0}, {1,1}, {0,1}};
        for (int i = 0; i < 4; ++i) source[i] = cv::Point2f(corners[2*i], corners[2*i+1]);
        auto matrix = cv::getPerspectiveTransform(source, paper);
        if (!cv::checkRange(matrix)) return false;
        std::copy_n(matrix.ptr<double>(), 9, output);
        return true;
    } catch (const std::exception&) { return false; }
}

bool DAWritingRegisterBGRA(DAWritingTracker *tracker, const uint8_t *bgra, int w, int h,
                           size_t stride, double output[9]) {
    if (!bgra || w <= 0 || h <= 0 || stride < size_t(w)*4) return false;
    try {
        cv::Mat gray;
        cv::cvtColor(cv::Mat(h, w, CV_8UC4, const_cast<uint8_t*>(bgra), stride), gray, cv::COLOR_BGRA2GRAY);
        return DAWritingRegister(tracker, gray.ptr<uint8_t>(), w, h, gray.step, output);
    } catch (const std::exception&) { return false; }
}
