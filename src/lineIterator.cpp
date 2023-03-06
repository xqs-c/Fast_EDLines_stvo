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

#include "lineIterator.h"

//STL
#include <cmath>
#include <utility>

//Implementation of Bresenham's line drawing Algorithm
//Bresenham线绘制算法的实现 
//Adapted from: https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#C.2B.2B

namespace StVO {

//构造函数
LineIterator::LineIterator(const double x1_, const double y1_, const double x2_, const double y2_)
    : x1(x1_), y1(y1_), x2(x2_), y2(y2_), steep(std::abs(y2_ - y1_) > std::abs(x2_ - x1_)) {

    //如果是陡直线，交换x,y轴，使之变为非陡直线
    if (steep) {
        std::swap(x1, y1);
        std::swap(x2, y2);
    }

    //如果起始点在结束点右边，交换起始点和结束点，使起始点始终在结束点左边
    if(x1 > x2) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }

    dx = x2 - x1;//x轴投影长度
    dy = std::abs(y2 - y1);//y轴投影长度

    error = dx / 2.0;  //
    ystep = (y1 < y2) ? 1 : -1;  //y方向是递增还是递减

    x = static_cast<int>(x1);  //向下取整
    y = static_cast<int>(y1);  //向下取整

    maxX = static_cast<int>(x2);  //x的最大值
}

//绘制下一个像素点
bool LineIterator::getNext(std::pair<int, int> &pixel) {

    if (x > maxX)
        return false;  //如果x大于最大值，说明已经绘制完毕

    //如果是陡直线
    if (steep)
        pixel = std::make_pair(y, x);  //交换x,y
    else
        pixel = std::make_pair(x, y);  

    error -= dy;
    if (error < 0) {
        y += ystep;
        error += dx;
    }

    x++;
    return true;
}

} // namespace StVO
