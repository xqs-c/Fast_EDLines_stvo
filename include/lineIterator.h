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
#include <utility>

namespace StVO {

//实现Bresenham线绘制算法
class LineIterator {
public:

    //构造函数
    LineIterator(const double x1_, const double y1_, const double x2_, const double y2_);

    //判断下一个直线上的像素点
    bool getNext(std::pair<int, int> &pixel);

private:

    const bool steep;  //判断线段是否是陡的，即y轴绝对差值大于x轴绝对差值
    double x1, y1, x2, y2;  //起始点（x1,y1) ,结束点（x2,y2)

    double dx, dy, error;  

    int maxX, ystep;
    int y, x;
};

} // namespace StVO
