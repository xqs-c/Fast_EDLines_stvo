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

#pragma once

//STL
#include <list>
#include <string>
#include <iomanip>

//OpenCV
#include <opencv2/core.hpp>

#include "pinholeStereoCamera.h"  //针孔双目相机头文件

namespace StVO {

class Dataset {
public:

    // Constructor  构造函数
    Dataset(const std::string &dataset_path, const PinholeStereoCamera &cam, int offset = 0, int nmax = 0, int step = 1);

    // Destrcutor
    virtual ~Dataset();

    // Reads next frame in the dataset sequence, returning true if image was successfully loaded
    //读取数据集序列中的下一帧，如果成功加载图像，则返回true
    bool nextFrame(cv::Mat &img_l, cv::Mat &img_r);

    // Returns if there are images still available in the sequence
    //返回序列中是否仍有图像可用
    inline bool hasNext();
    
    //获取数据集的时间戳
    void getTimeStamps(const std::string &datasetTime_path,int offset = 0, int nmax = 0, int step = 1);
    
    //存放EuRoC数据集的时间戳
    vector<double>  vTimeStamps;

private:

    std::list<std::string> images_l, images_r;  //双目图像帧列表
   // int images_size;
    const PinholeStereoCamera &cam;  //常量指针，指向针孔相机
    
};

} // namespace StVO

