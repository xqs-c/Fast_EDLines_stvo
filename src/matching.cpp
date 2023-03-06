/*****************************************************************************
**      Stereo VO and SLAM by combining point and line segment features     **
******************************************************************************
**                                                                          **
**  Copyright(c) 2016-2018, Ruben Gomez-Ojeda, University of Malaga         **
**  Copyright(c) 2016-2018, David Zuñiga-Noël, University of Malaga         **
**  Copyright(c) 2016-2018, MAPIR group, University of Malaga               **
**                                                                          **
**  This program is free software: you can redistribute it and/or modify    **
**  it under the terms of the GNU General Public License (version 3) as     **
**  published by the Free Software Foundation.                              **
**                                                                          **
**  This program is distributed in the hope that it will be useful, but     **
**  WITHOUT ANY WARRANTY; without even the implied warranty of              **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            **
**  GNU General Public License for more details.                            **
**                                                                          **
**  You should have received a copy of the GNU General Public License       **
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.   **
**                                                                          **
*****************************************************************************/

#include "matching.h"

//STL
#include <cmath>
#include <functional>
#include <future>
#include <limits>
#include <stdexcept>

//OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "config.h"
#include "gridStructure.h"

namespace StVO {
    
//帧间匹配
//输入：上一帧的左目特征点的描述子 当前帧的左目特征点的描述子  最佳距离和次佳距离比值的阈值  匹配对容器
int matchNNR(const cv::Mat &desc1, const cv::Mat &desc2, float nnr, std::vector<int> &matches_12) {

    int matches = 0; //匹配点对数量
    matches_12.resize(desc1.rows, -1);//初始化匹配点对容器

    std::vector<std::vector<cv::DMatch>> matches_; //
    cv::Ptr<cv::BFMatcher> bfm = cv::BFMatcher::create(cv::NORM_HAMMING, false); // cross-check  创建一个ORB特征点匹配器
    //cv::Ptr<cv::DescriptorMatcher> bfm = cv::BFMatcher::create("cv::NORM_HAMMING");
    bfm->knnMatch(desc1, desc2, matches_, 2);//进行特征匹配 暴力匹配

    if (desc1.rows != matches_.size())
        throw std::runtime_error("[matchNNR] Different size for matches and descriptors!"); //如果上一帧描述子的数量不等于最后匹配对的数量，则输出错误信息

    //遍历每一对匹配点对
    for (int idx = 0; idx < desc1.rows; ++idx) {
        //如果最佳距离与次佳距离的比值小于阈值
        if (matches_[idx][0].distance < matches_[idx][1].distance * nnr) {
            matches_12[idx] = matches_[idx][0].trainIdx; //将匹配的当前帧的特征点索引存入匹配容器中
            matches++;//匹配数量加1
        }
    }

    return matches;//返回匹配点对数量
}

//帧间点特征的匹配 是否开启双线程正反帧间匹配，并剔除正反匹配不一致的匹配点对
//输入：上一帧的左目特征点的描述子 当前帧的左目特征点的描述子  最佳距离和次佳距离比值的阈值  匹配对容器
int match(const cv::Mat &desc1, const cv::Mat &desc2, float nnr, std::vector<int> &matches_12) {

    //如果正反匹配标志位置一
    if (Config::bestLRMatches()) {
        int matches;  //成功匹配数量
        std::vector<int> matches_21;  //创建一个反匹配容器
        //如果并行正反匹配标志位置一 
        if (Config::lrInParallel()) {
            auto match_12 = std::async(std::launch::async, &matchNNR,
                                  std::cref(desc1), std::cref(desc2), nnr, std::ref(matches_12)); //创建一个新线程，进行正匹配，将上一帧的特征点与当前帧的特征点匹配
            auto match_21 = std::async(std::launch::async, &matchNNR,
                                  std::cref(desc2), std::cref(desc1), nnr, std::ref(matches_21));//创建一个新线程，进行反匹配，将当前帧的特征点与上一帧帧的特征点匹配
            match_12.wait();//TODO （修改过，源代码没有这行代码）
            matches = match_12.get();
            match_21.wait();
        } else {
            matches = matchNNR(desc1, desc2, nnr, matches_12);
            matchNNR(desc2, desc1, nnr, matches_21);
        }

        //遍历匹配对
        for (int i1 = 0; i1 < matches_12.size(); ++i1) {
            int &i2 = matches_12[i1];  //取出匹配的当前帧特征点
            //如果当前帧特征点索引存在并且与当前帧特征点最佳匹配的特征点与上一帧特征点不是同一个点，即不是互为最佳匹配点，则剔除该对匹配
            if (i2 >= 0 && matches_21[i2] != i1) {
                i2 = -1;
                matches--;
            }
        }

        return matches; //返回正确匹配点对数量
    } 
    else
        return matchNNR(desc1, desc2, nnr, matches_12);
}

//计算描述子之间的汉明距离
int distance(const cv::Mat &a, const cv::Mat &b) {

    // adapted from: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;
    for(int i = 0; i < 8; i++, pa++, pb++) {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

//点特征网格匹配，通过网格进行搜索，加快匹配速度，匹配结果存放到matches_12中，返回成功匹配点对数量 双目匹配
int matchGrid(const std::vector<point_2d> &points1, const cv::Mat &desc1, const GridStructure &grid, const cv::Mat &desc2, const GridWindow &w, std::vector<int> &matches_12) {

    if (points1.size() != desc1.rows)
        throw std::runtime_error("[matchGrid] Each point needs a corresponding descriptor!");  //如果左目图像的特征点与特征点的描述子数量不一致，输出错误信息

    int matches = 0;  //匹配的点对的数量
    matches_12.resize(desc1.rows, -1);  //初始化

    int best_d, best_d2, best_idx;  //定义最佳距离，次佳距离，最佳匹配的索引
    std::vector<int> matches_21, distances; //定义左目图像中与右目特征点相匹配的特征点在左目图像中的索引容器，以及两个描述子的汉明距离

    //如果双重匹配（正反匹配）标志位置位，则初始化左目匹配容器
    if (Config::bestLRMatches()) {
        matches_21.resize(desc2.rows, -1);
        distances.resize(desc2.rows, std::numeric_limits<int>::max());
    }

    //遍历左目图像的特征点
    for (int i1 = 0; i1 < points1.size(); ++i1) {

        best_d = std::numeric_limits<int>::max();
        best_d2 = std::numeric_limits<int>::max();
        best_idx = -1;//初始化

        const std::pair<int, int> &coords = points1[i1];  //取出该特征点所在网格的索引
        cv::Mat desc = desc1.row(i1);  //取出描述子

        std::unordered_set<int> candidates;  //创建存放候选匹配点索引的容器
        grid.get(coords.first, coords.second, w, candidates);  //获取对应网格中右目图像特征点的坐标索引，存放到候选匹配点容器中

        if (candidates.empty()) continue;  //如果候选匹配点容器为空，跳过当前特征点
        //遍历每一个候选特征点
        for (const int &i2 : candidates) {
            if (i2 < 0 || i2 >= desc2.rows) continue;  //跳过不合法的候选匹配点

            const int d = distance(desc, desc2.row(i2));  //计算两个描述子的汉明距离

            //判断是否也是右目的最佳匹配
            if (Config::bestLRMatches()) {
                if (d < distances[i2]) {
                    distances[i2] = d;
                    matches_21[i2] = i1;
                } else continue;
            }

            if (d < best_d) {
                best_d2 = best_d;
                best_d = d;
                best_idx = i2;
            } else if (d < best_d2)
                best_d2 = d;
        }

        //判断最小汉明距离与次小汉明距离的比值是否小于阈值
        if (best_d < best_d2 * Config::minRatio12P()) {
            matches_12[i1] = best_idx;
            matches++;
        }
    }

    //如果正反匹配标志位置一
    if (Config::bestLRMatches()) {
        //遍历每一对匹配点对
        for (int i1 = 0; i1 < matches_12.size(); ++i1) {
            int &i2 = matches_12[i1];  //取出匹配的右目图像特征点的索引
            //剔除正反匹配不一致的匹配点对
            if (i2 >= 0 && matches_21[i2] != i1) {
                i2 = -1;
                matches--;
            }
        }
    }

    return matches;
}

//线特征网格匹配 双目匹配
//lines1：左目线特征两端点所在网格的索引  desc1：左目线特征的描述子  grid：右目线特征每个网格中右目线特征的索引  desc2：右目线特征描述子  directions2：右目线特征的方向
int matchGrid(const std::vector<line_2d> &lines1, const cv::Mat &desc1,
              const GridStructure &grid, const cv::Mat &desc2, const std::vector<std::pair<double, double>> &directions2,
              const GridWindow &w,
              std::vector<int> &matches_12) {

    if (lines1.size() != desc1.rows)  //如果左目线特征数量与描述子数量不相等，输出错误信息
        throw std::runtime_error("[matchGrid] Each line needs a corresponding descriptor!");

    int matches = 0;  //匹配对数量初始化为0
    matches_12.resize(desc1.rows, -1);  //匹配对容器初始化

    int best_d, best_d2, best_idx;  //定义最佳距离、次佳距离、最佳匹配的右目特征索引
    std::vector<int> matches_21, distances;  //定义反匹配容器和距离容器

    //如果正反匹配标志位置一
    if (Config::bestLRMatches()) {
        matches_21.resize(desc2.rows, -1); //反匹配容器初始化
        distances.resize(desc2.rows, std::numeric_limits<int>::max());  //距离容器初始化
    }

    //遍历左目图像的线特征，进行双目匹配
    for (int i1 = 0; i1 < lines1.size(); ++i1) {

        best_d = std::numeric_limits<int>::max();  //初始化最佳距离
        best_d2 = std::numeric_limits<int>::max();  //初始化次佳距离
        best_idx = -1;  //初始化最佳匹配右目索引

        const line_2d &coords = lines1[i1]; //取出左目特征端点所在的网格索引
        cv::Mat desc = desc1.row(i1); //取出左目线特征的描述子

        const point_2d sp = coords.first;  //左目线特征起始点所在的网格索引
        const point_2d ep = coords.second;  //左目线特征结束点所在的网格索引

        std::pair<double, double> v = std::make_pair(ep.first - sp.first, ep.second - sp.second);   //左目线特征两端点的方向
        normalize(v);  //归一化

        std::unordered_set<int> candidates;  //候选匹配线特征的索引
        grid.get(sp.first, sp.second, w, candidates);  //获取左目起始点所在网格内所有的端点作为候选点
        grid.get(ep.first, ep.second, w, candidates);  //获取左目结束点所在网格内所有的端点作为候选点

        if (candidates.empty()) continue;  //如果候选匹配点容器为空，则跳过该线特征
        //遍历每一个候选线特征端点
        for (const int &i2 : candidates) {
            if (i2 < 0 || i2 >= desc2.rows) continue;  //如果端点索引非法，则跳过该线特征端点

            if (std::abs(dot(v, directions2[i2])) < Config::lineSimTh())
                continue; //判断左目特征与待匹配的右目特征的方向夹角是否大于阈值，如果大于则跳过

            const int d = distance(desc, desc2.row(i2));  //计算两个线特征描述子的汉明距离

            //如果正反匹配标志位置一，记录反匹配的最佳距离和最佳匹配索引
            if (Config::bestLRMatches()) {
                if (d < distances[i2]) {
                    distances[i2] = d;
                    matches_21[i2] = i1;
                } else continue;
            }

            //如果距离小于最佳距离，更新最佳距离、次佳距离和最佳匹配的索引
            if (d < best_d) {
                best_d2 = best_d;
                best_d = d;
                best_idx = i2;
            } else if (d < best_d2)
                best_d2 = d;  //如果距离小于次佳距离大于最佳距离，则更新次佳距离
        }

        //判断最佳距离与次佳距离的比值是否小于阈值，若是，则将最佳匹配索引存入最佳匹配索引容器，匹配数量加1
        if (best_d < best_d2 * Config::minRatio12P()) {
            matches_12[i1] = best_idx;
            matches++;
        }
    }

    //如果正反匹配标志位置一，判断是否是相互的最佳匹配，如果不是则剔除该对匹配
    if (Config::bestLRMatches()) {
        for (int i1 = 0; i1 < matches_12.size(); ++i1) {
            int &i2 = matches_12[i1];
            if (i2 >= 0 && matches_21[i2] != i1) {
                i2 = -1;
                matches--;
            }
        }
    }

    return matches;  //最终返回正确匹配对数
}

} //namesapce StVO
