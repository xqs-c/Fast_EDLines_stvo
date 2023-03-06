
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

#include <stereoFrame.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

#include "lineIterator.h"
#include "matching.h"
#include "EDLib.h"
#include <chrono>
//#include "ED.h"
//#include "EDPF.h"
//#include "EDLines.h"
//#include <../../EDLines_my/LineDescriptor.hh>

namespace StVO{

/* Constructor and main method */

StereoFrame::StereoFrame(){}

StereoFrame::StereoFrame(const Mat img_l_, const Mat img_r_ , const int idx_, PinholeStereoCamera *cam_,ORBextractor* extractor_left,ORBextractor* extractor_right,double prev_error,double pprev_error) :
    img_l(img_l_), img_r(img_r_), frame_idx(idx_), cam(cam_) ,
    mpORBextractorLeft(extractor_left),mpORBextractorRight(extractor_right),
    prev_frame_err_norm(prev_error),pprev_frame_err_norm(pprev_error){

    if (img_l_.size != img_r_.size)
        throw std::runtime_error("[StereoFrame] Left and right images have different sizes");

    inv_width  = GRID_COLS / static_cast<double>(img_l.cols); //每个网格宽的倒数
    inv_height = GRID_ROWS / static_cast<double>(img_l.rows);  //每个网格高的倒数
    
    
}

StereoFrame::~StereoFrame()
{
    for( auto pt: stereo_pt )
        delete pt;
    for( auto ls: stereo_ls )
        delete ls;
    //delete mpORBextractor;
}

//提取双目特征
void StereoFrame::extractStereoFeatures( double llength_th, int fast_th )
{
    if(Config::useTexture())
    {
        //采取场景纹理信息丰富度线特征提取策略
        detectStereoPoints(fast_th);  //提取双目点特征，并得到双目点特征匹配
        lines_l.clear();   //清空线特征容器
        lines_r.clear();
        stereo_ls.clear();
        
        double f_x=0.0; 
        
         string fPath="f_x.txt";
         ofstream of;
         of.open(fPath.c_str(),ios::out | ios::app);
        
        //如果提取到的双目点特征数量小于600，说明特征数量不足，位于低纹理场景，应提取线特征
        //或者上一帧位姿优化失败，说明需要更多的特征
        if( stereo_pt.size() < Config::fuseLineth() || prev_frame_err_norm == -1 || pprev_frame_err_norm == -1)
        {
            f_x=1.0;
        }
        else
        {
            f_x=0.5*(double(Config::fuseLineth())/double(stereo_pt.size()))+0.3*prev_frame_err_norm+0.2*pprev_frame_err_norm;
        }
        
         of<<f_x<<endl;
         of.close();
        
        if(f_x>=0.45)
        {
             //cout<<"提取到双目点特征数量为："<<stereo_pt.size()<<endl;
            detectStereoLineSegments(llength_th);
        }
    }
    else
    {
        //同时提取点线特征
        if( Config::plInParallel() )  //如果双线程提取点线特征标志位为true，则开启双线程
        {
            auto detect_p = async(launch::async, &StereoFrame::detectStereoPoints,        this, fast_th );
            auto detect_l = async(launch::async, &StereoFrame::detectStereoLineSegments,  this, llength_th );
            detect_p.wait();
            detect_l.wait();
        }
        else
        {
            detectStereoPoints(fast_th);
            detectStereoLineSegments(llength_th);
        } 
    }
}

/* Stereo point features extraction */

void StereoFrame::detectStereoPoints( int fast_th )
{

    if( !Config::hasPoints() )  //如果不提取点特征，返回
        return;

    // detect and estimate each descriptor for both the left and right image
    if( Config::lrInParallel() ) //并行提取左右图像的点特征
    {
        auto detect_l = async(launch::async, &StereoFrame::detectPointFeaturesLeft, this, img_l, ref(points_l), ref(pdesc_l), fast_th );
        auto detect_r = async(launch::async, &StereoFrame::detectPointFeaturesRight, this, img_r, ref(points_r), ref(pdesc_r), fast_th );
        detect_l.wait();
        detect_r.wait();
    }
    else
    {
        detectPointFeaturesLeft( img_l, points_l, pdesc_l, fast_th );  //提取左目点特征
        detectPointFeaturesRight( img_r, points_r, pdesc_r, fast_th );  //提取右目点特征
    }

    // perform the stereo matching
    matchStereoPoints(points_l, points_r, pdesc_l, pdesc_r, (frame_idx==0) );  //执行双目匹配，并完成路标点的三角化，得到特征点对应的相机坐标

}


//提取点特征
//这里采用opencv自带的点特征提取器
//可以将图像分块提取特征点，然后加入四叉树筛选特征点，使特征点提取更加均匀
void StereoFrame::detectPointFeaturesLeft( Mat img, vector<KeyPoint> &points, Mat &pdesc, int fast_th )
{
    // Detect point features
    if( Config::hasPoints() )
    {
        if(Config::useMyORB()) //使用自己的ORB提取器
        {
            // 这里使用了仿函数来完成，重载了括号运算符 ORBextractor::operator() 
            (*mpORBextractorLeft)(img,				//待提取特征点的图像
                            cv::Mat(),		//掩摸图像, 实际没有用到
                            points,			//输出变量，用于保存提取后的特征点
                            pdesc);	//输出变量，用于保存特征点的描述子
        }
        else{
            //cout<<"提取点特征"<<endl;
            int fast_th_ = Config::orbFastTh();  //fast角点的阈值
            if( fast_th != 0 )
                fast_th_ = fast_th;
            Ptr<ORB> orb = ORB::create( Config::orbNFeatures(), Config::orbScaleFactor(), Config::orbNLevels(),
                                        Config::orbEdgeTh(), 0, Config::orbWtaK(), Config::orbScore(),
                                        Config::orbPatchSize(), fast_th_ );  //创建一个orb特征提取器
            /*
            * Config::orbNFeatures() ：最多提取的特征点的数量；
            * Config::orbScaleFactor()：金字塔图像之间的尺度参数；
            * Config::orbNLevels()：高斯金字塔的层数；
            * Config::orbEdgeTh()：边缘阈值，这个值主要是根据后面的patchSize来定的，靠近边缘edgeThreshold以内的像素是不检测特征点的；
            * Config::orbWtaK()：用于产生BIREF描述子的点对的个数，一般为2个，也可以设置为3个或4个，那么这时候描述子之间的距离计算就不能用汉明距离了，
            *                   而是应该用一个变种。OpenCV中，如果设置WET_K = 2，则选用点对就只有2个点，匹配的时候距离参数选择NORM_HAMMING，
            *                   如果WET_K设置为3或4，则BIREF描述子会选择3个或4个点，那么后面匹配的时候应该选择的距离参数为NORM_HAMMING2；
            * Config::orbScore()：用于对特征点进行排序的算法，你可以选择HARRIS_SCORE，也可以选择FAST_SCORE，但是它也只是比前者快一点点而已；
            * Config::orbPatchSize()：用于计算BIREF描述子的特征点邻域大小。
            */
            orb->detectAndCompute( img, Mat(), points, pdesc, false);  //point特征点容器，pdesc特征点对应的描述子容器
        }      
    }
}

//可以将图像分块提取特征点，然后加入四叉树筛选特征点，使特征点提取更加均匀
void StereoFrame::detectPointFeaturesRight( Mat img, vector<KeyPoint> &points, Mat &pdesc, int fast_th )
{
    // Detect point features
    if( Config::hasPoints() )
    {
        if(Config::useMyORB()) //使用自己的ORB提取器
        {
            // 这里使用了仿函数来完成，重载了括号运算符 ORBextractor::operator() 
            (*mpORBextractorRight)(img,				//待提取特征点的图像
                            cv::Mat(),		//掩摸图像, 实际没有用到
                            points,			//输出变量，用于保存提取后的特征点
                            pdesc);	//输出变量，用于保存特征点的描述子
        }
        else{
            //cout<<"提取点特征"<<endl;
            int fast_th_ = Config::orbFastTh();  //fast角点的阈值
            if( fast_th != 0 )
                fast_th_ = fast_th;
            Ptr<ORB> orb = ORB::create( Config::orbNFeatures(), Config::orbScaleFactor(), Config::orbNLevels(),
                                        Config::orbEdgeTh(), 0, Config::orbWtaK(), Config::orbScore(),
                                        Config::orbPatchSize(), fast_th_ );  //创建一个orb特征提取器
            /*
            * Config::orbNFeatures() ：最多提取的特征点的数量；
            * Config::orbScaleFactor()：金字塔图像之间的尺度参数；
            * Config::orbNLevels()：高斯金字塔的层数；
            * Config::orbEdgeTh()：边缘阈值，这个值主要是根据后面的patchSize来定的，靠近边缘edgeThreshold以内的像素是不检测特征点的；
            * Config::orbWtaK()：用于产生BIREF描述子的点对的个数，一般为2个，也可以设置为3个或4个，那么这时候描述子之间的距离计算就不能用汉明距离了，
            *                   而是应该用一个变种。OpenCV中，如果设置WET_K = 2，则选用点对就只有2个点，匹配的时候距离参数选择NORM_HAMMING，
            *                   如果WET_K设置为3或4，则BIREF描述子会选择3个或4个点，那么后面匹配的时候应该选择的距离参数为NORM_HAMMING2；
            * Config::orbScore()：用于对特征点进行排序的算法，你可以选择HARRIS_SCORE，也可以选择FAST_SCORE，但是它也只是比前者快一点点而已；
            * Config::orbPatchSize()：用于计算BIREF描述子的特征点邻域大小。
            */
            orb->detectAndCompute( img, Mat(), points, pdesc, false);  //point特征点容器，pdesc特征点对应的描述子容器
        }      
    }
}

//匹配双目点特征
void StereoFrame::matchStereoPoints( vector<KeyPoint> points_l, vector<KeyPoint> points_r, Mat &pdesc_l_, Mat pdesc_r, bool initial )
{

    // Points stereo matching
    // --------------------------------------------------------------------------------------------------------------------
    stereo_pt.clear();
    if (!Config::hasPoints() || points_l.empty() || points_r.empty())
        return;  //如果没有提取点特征，左目点特征为空或者右目特征点为空，返回

    std::vector<point_2d> coords;  //左目特征点所在网格的索引，如在第（2，1）个网格即第2行第1列个网格
    coords.reserve(points_l.size());
    for (const KeyPoint &kp : points_l)  //遍历每一个左目特征点
        coords.push_back(std::make_pair(kp.pt.x * inv_width, kp.pt.y * inv_height));  //得到网格的索引

    //Fill in grid
    GridStructure grid(GRID_ROWS, GRID_COLS);
    //遍历右目特征点
    for (int idx = 0; idx < points_r.size(); ++idx) {
        const KeyPoint &kp = points_r[idx];//取出右目特征点
        grid.at(kp.pt.x * inv_width, kp.pt.y * inv_height).push_back(idx);//保存每个网格中右目特征点的索引
    }

    GridWindow w;
    w.width = std::make_pair(Config::matchingSWs(), 0);
    w.height = std::make_pair(0, 0);

    std::vector<int> matches_12; //存放右目图像中与左目图像配对的特征点的索引
    //调用网格匹配函数，得到匹配结果存放在matches_12中，并返回成功匹配的点对数量
    //1、两点特征的描述子的最佳距离与次佳距离的比值是否小于阈值
    //2、两点特征的描述子距离是否是彼此的最佳距离
    matchGrid(coords, pdesc_l, grid, pdesc_r, w, matches_12);  
    //match(pdesc_l, pdesc_r, Config::minRatio12P(), matches_12);

    // bucle around pmatches
    Mat pdesc_l_aux;
    int pt_idx = 0; 
    //遍历所有匹配成功的点对
    for (int i1 = 0; i1 < matches_12.size(); ++i1) {
        const int i2 = matches_12[i1];  //取出匹配点在右目图像中的索引
        if (i2 < 0) continue;  //如果该左目特征点的右目匹配点不存在，则跳过

        // check stereo epipolar constraint  检查双目极线约束
        //如果匹配点对在y轴方向的距离小于给定阈值 
        if (std::abs(points_l[i1].pt.y - points_r[i2].pt.y) <= Config::maxDistEpip()) {
            // check minimal disparity
            double disp_ = points_l[i1].pt.x - points_r[i2].pt.x;  //计算匹配点对在x轴方向上的距离
            //如果距离大于给定阈值 视差足够大
            if (disp_ >= Config::minDisp()){
                pdesc_l_aux.push_back(pdesc_l_.row(i1));  //将该左目特征点的描述子存入容器
                Vector2d pl_(points_l[i1].pt.x, points_l[i1].pt.y);  //取出该左目特征点的像素坐标
                Vector3d P_ = cam->backProjection(pl_(0), pl_(1), disp_);  //将该左目特征点投影到相机平面
                if (initial) //如果是第一帧
                    stereo_pt.push_back(new PointFeature(pl_, disp_, P_, pt_idx++, points_l[i1].octave));  //保存该点特征
                else
                    stereo_pt.push_back(new PointFeature(pl_, disp_, P_, -1, points_l[i1].octave));
            }
        }
    }

    pdesc_l_ = pdesc_l_aux;  //保存剔除不满足极线约束的特征点后的描述子容器
}

void StereoFrame::matchPointFeatures(BFMatcher* bfm, Mat pdesc_1, Mat pdesc_2, vector<vector<DMatch>> &pmatches_12  )
{
    bfm->knnMatch( pdesc_1, pdesc_2, pmatches_12, 2);
}

/* Stereo line segment features extraction */
//双目线特征提取器
void StereoFrame::detectStereoLineSegments(double llength_th)
{
    //如果不提取线特征，则返回
    if( !Config::hasLines() )
        return;

    // detect and estimate each descriptor for both the left and right image
    //并行提取左右目图像的线特征
    if( Config::lrInParallel() )
    {
        auto detect_l = async(launch::async, &StereoFrame::detectLineFeatures, this, img_l, ref(lines_l), ref(ldesc_l), llength_th );
        auto detect_r = async(launch::async, &StereoFrame::detectLineFeatures, this, img_r, ref(lines_r), ref(ldesc_r), llength_th );
        detect_l.wait();
        detect_r.wait();
    }
    else
    {
        detectLineFeatures( img_l, lines_l, ldesc_l, llength_th );  //提取LSD线特征并按照线特征的response从大到小排列，计算线特征的描述子
        detectLineFeatures( img_r, lines_r, ldesc_r, llength_th );
    }

    // perform the stereo matching
    //执行线特征双目匹配
    matchStereoLines(lines_l,  lines_r,  ldesc_l, ldesc_r, (frame_idx==0));

}


//线特征提取
// img：输入图像  lines：线特征存放容器  ldesc：线特征对应的描述子  min_line_length：线特征最短长度阈值
void StereoFrame::detectLineFeatures( Mat img, vector<KeyLine> &lines, Mat &ldesc, double min_line_length )
{
    //保存每帧线特征提取时间和线特征提取数量
//     string timePath="ftime.txt";
//     string linesPath="linescount.txt";
//     ofstream oft;
//     ofstream ofl;
//     oft.open(timePath.c_str(),ios::out | ios::app);
//     ofl.open(linesPath.c_str(),ios::out | ios::app);

    // Detect line features
    lines.clear();   //清空线特征容器
    Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();  //LBD线特征描述子
    //如果提取线特征
    if( Config::hasLines() )
    {
        //如果使用LSD线特征提取器
        if( Config::useLSDLines() )
        {
            Ptr<line_descriptor::LSDDetectorC> lsd = line_descriptor::LSDDetectorC::createLSDDetectorC();  //创建一个LSD线特征提取器
            // lsd parameters
            //LSD线特征参数
            line_descriptor::LSDDetectorC::LSDOptions opts;
            opts.refine       = Config::lsdRefine();
            opts.scale        = Config::lsdScale();  //用于检测线段的图像比例 
            opts.sigma_scale  = Config::lsdSigmaScale(); //高斯滤波器的sigma
            opts.quant        = Config::lsdQuant();  //限制于梯度范数上的量化误差
            opts.ang_th       = Config::lsdAngTh();  //梯度角度公差（度）
            opts.log_eps      = Config::lsdLogEps();  //检测阈值（仅用于高级细化）
            opts.density_th   = Config::lsdDensityTh();  //封闭矩形中对齐区域点的最小密度
            opts.n_bins       = Config::lsdNBins();  //梯度模伪序中的面元数
            opts.min_length   = min_line_length;  //线段的最小长度
            //chrono::steady_clock::time_point t1=chrono::steady_clock::now();//记录开始时间
            lsd->detect( img, lines, Config::lsdScale(), 1, opts); //提取线特征
//             chrono::steady_clock::time_point t2=chrono::steady_clock::now();//记录结束时间
//             chrono::duration<double> time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
//             oft<<"time LSD extraction: "<<time_used.count()<<"s"<<std::endl;
//             ofl<<"lines number: "<<lines.size()<<endl;
//             oft.close();
//             ofl.close();
//             vector<KeyLine> v_lines;
//             lsd->detect( img, v_lines, Config::lsdScale(), 1, opts); //提取线特征
//             
//             //融合线特征
//             fuseStereoLines(img,v_lines);
//             
//             for(vector<KeyLine>::iterator it=v_lines.begin();it!=v_lines.end();it++)
//             {
//                 if(it->class_id!=-1)
//                 {
//                     lines.push_back(*it);
//                 }
//             }
//         
//             //重新分配id
//             for(int i=0;i<lines.size();i++)
//             {
//                 lines[i].class_id=i;
//             }
            
            // filter lines
            //筛选线特征
            //如果提取到的线特征数量大于线特征数量阈值，并且线特征数量阈值不等于0
            if( lines.size()>Config::lsdNFeatures() && Config::lsdNFeatures()!=0  )
            {
                // sort lines by their response
                //根据线特征的response进行排序
                sort( lines.begin(), lines.end(), sort_lines_by_response() ); //按从大到小排列
                //sort( lines.begin(), lines.end(), sort_lines_by_length() );  //按线特征的长度从大到小排列
                lines.resize(Config::lsdNFeatures());  //剔除多余的线特征
                // reassign index
                //重新分配线特征的索引
                for( int i = 0; i < Config::lsdNFeatures(); i++  )
                    lines[i].class_id = i;
            }
            //计算线特征的描述子
            lbd->compute( img, lines, ldesc);
        }
        else if(Config::useFLDLines()) //使用FLD线特征提取器
        {
            Mat fld_img, img_gray;
            vector<Vec4f> fld_lines;

            if( img.channels() != 1 )
            {
                cv::cvtColor( img, img_gray, CV_RGB2GRAY );
                img_gray.convertTo( fld_img, CV_8UC1 );
            }
            else
                img.convertTo( fld_img, CV_8UC1 );

            Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector(min_line_length);
           // chrono::steady_clock::time_point t1=chrono::steady_clock::now();//记录开始时间
            fld->detect( fld_img, fld_lines );
//             chrono::steady_clock::time_point t2=chrono::steady_clock::now();//记录结束时间
//             chrono::duration<double> time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
//             oft<<"time FLD extraction: "<<time_used.count()<<"s"<<std::endl;
//             ofl<<"lines number: "<<fld_lines.size()<<endl;
//             oft.close();
//             ofl.close();

            // filter lines
            if( fld_lines.size()>Config::lsdNFeatures() && Config::lsdNFeatures()!=0  )
            {
                // sort lines by their response
                sort( fld_lines.begin(), fld_lines.end(), sort_flines_by_length() );
                fld_lines.resize(Config::lsdNFeatures());
            }

            // loop over lines object transforming into a vector<KeyLine>
            lines.reserve(fld_lines.size());
            for( int i = 0; i < fld_lines.size(); i++ )
            {
                KeyLine kl;
                double octaveScale = 1.f;
                int    octaveIdx   = 0;

                kl.startPointX     = fld_lines[i][0] * octaveScale;
                kl.startPointY     = fld_lines[i][1] * octaveScale;
                kl.endPointX       = fld_lines[i][2] * octaveScale;
                kl.endPointY       = fld_lines[i][3] * octaveScale;

                kl.sPointInOctaveX = fld_lines[i][0];
                kl.sPointInOctaveY = fld_lines[i][1];
                kl.ePointInOctaveX = fld_lines[i][2];
                kl.ePointInOctaveY = fld_lines[i][3];

                kl.lineLength = (float) sqrt( pow( fld_lines[i][0] - fld_lines[i][2], 2 ) + pow( fld_lines[i][1] - fld_lines[i][3], 2 ) );

                kl.angle    = atan2( ( kl.endPointY - kl.startPointY ), ( kl.endPointX - kl.startPointX ) );
                kl.class_id = i;
                kl.octave   = octaveIdx;
                kl.size     = ( kl.endPointX - kl.startPointX ) * ( kl.endPointY - kl.startPointY );
                kl.pt       = Point2f( ( kl.endPointX + kl.startPointX ) / 2, ( kl.endPointY + kl.startPointY ) / 2 );

                kl.response = kl.lineLength / max( fld_img.cols, fld_img.rows );
                cv::LineIterator li( fld_img, Point2f( fld_lines[i][0], fld_lines[i][1] ), Point2f( fld_lines[i][2], fld_lines[i][3] ) );
                kl.numOfPixels = li.count;

                lines.push_back( kl );

            }

            // compute lbd descriptor
            lbd->compute( fld_img, lines, ldesc);
        }
        else if(Config::useEDLines())//TODO 使用EDLines
        {
             Mat EDlines_img, img_gray;
             //vector<Vec4f> EDlines;  //保存提取的EDLines线段容器
 
             //读取图像
             if( img.channels() != 1 ) //如果原始图像是RGB图像，转化为灰度图
             {
                 cv::cvtColor( img, img_gray, CV_RGB2GRAY );
                 img_gray.convertTo( EDlines_img, CV_8UC1 );
             }
             else
                 img.convertTo( EDlines_img, CV_8UC1 );
             
            //chrono::steady_clock::time_point t1=chrono::steady_clock::now();//记录开始时间
             //采用无参边缘检测算法
            EDPF testEDPF = EDPF(EDlines_img);
             
             //进行线特征提取
            EDLines testEDLines = EDLines(testEDPF);
//             chrono::steady_clock::time_point t2=chrono::steady_clock::now();//记录结束时间
//             chrono::duration<double> time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
//             oft<<"time EDLines extraction: "<<time_used.count()<<"s"<<std::endl;
//             oft.close();
           
              
              //获取线特征信息
            vector<LS> EDlines = testEDLines.getLines();
            /*
            ofl<<"lines number: "<<EDlines.size()<<endl;
            ofl.close();*/
//             int noLines = testEDLines.getLinesNo();
//             cout << "Number of line segments: " << noLines << endl;
              
            // loop over lines object transforming into a vector<KeyLine>
            //遍历线特征对象转换为向量<KeyLine>
            vector<KeyLine> v_lines;
            v_lines.reserve(EDlines.size());
            for( int i = 0; i < EDlines.size(); i++ )
            {
                KeyLine kl;
                double octaveScale = 1.f;
                int    octaveIdx   = 0;

                kl.startPointX     = EDlines[i].start.x * octaveScale;
                kl.startPointY     = EDlines[i].start.y * octaveScale;
                kl.endPointX       = EDlines[i].end.x * octaveScale;
                kl.endPointY       = EDlines[i].end.y * octaveScale;

                kl.sPointInOctaveX = EDlines[i].start.x;
                kl.sPointInOctaveY = EDlines[i].start.y;
                kl.ePointInOctaveX = EDlines[i].end.x;
                kl.ePointInOctaveY = EDlines[i].end.y;

                kl.lineLength = (float) sqrt( pow( EDlines[i].start.x - EDlines[i].end.x, 2 ) + pow( EDlines[i].start.y - EDlines[i].end.y, 2 ) );

                kl.angle    = atan2( ( kl.endPointY - kl.startPointY ), ( kl.endPointX - kl.startPointX ) );
                kl.class_id = i;
                kl.octave   = octaveIdx;
                kl.size     = ( kl.endPointX - kl.startPointX ) * ( kl.endPointY - kl.startPointY );
                kl.pt       = Point2f( ( kl.endPointX + kl.startPointX ) / 2, ( kl.endPointY + kl.startPointY ) / 2 );

                kl.response = kl.lineLength / max( EDlines_img.cols, EDlines_img.rows );
                cv::LineIterator li( EDlines_img, Point2f( EDlines[i].start.x, EDlines[i].start.y ), Point2f( EDlines[i].end.x, EDlines[i].end.y ) );
                kl.numOfPixels = li.count;

                v_lines.push_back( kl );

            }
            
             sort( v_lines.begin(), v_lines.end(), sort_lines_by_response() ); //按从大到小排列
            //融合断裂或遮挡的线特征
             //************************** 将提取到的线特征进行融合 *************************
            //fuseStereoLines(EDlines_img,v_lines);
            
            
            //************************** 将融合后的线特征进行整理，删除多余的线特征，并重新分配id *************************
            lines.reserve(v_lines.size());
            for(vector<KeyLine>::iterator it=v_lines.begin();it!=v_lines.end();it++)
            {
                if(it->class_id!=-1)
                {
                    lines.push_back(*it);
                }
            }
        
            //重新分配id
            for(int i=0;i<lines.size();i++)
            {
                lines[i].class_id=i;
            }
            
            // filter lines
            //筛选线特征
            //如果提取到的线特征数量大于线特征数量阈值，并且线特征数量阈值不等于0
            if( lines.size()>Config::lsdNFeatures() && Config::lsdNFeatures()!=0  )
            {
                // sort lines by their response
                //根据线特征的response进行排序
                sort( lines.begin(), lines.end(), sort_lines_by_response() ); //按从大到小排列
                //sort( lines.begin(), lines.end(), sort_lines_by_length() );  //按线特征的长度从大到小排列
                lines.resize(Config::lsdNFeatures());  //剔除多余的线特征
                // reassign index
                //重新分配线特征的索引
                for( int i = 0; i < Config::lsdNFeatures(); i++  )
                    lines[i].class_id = i;
            }

            //cout<<"筛选后线特征数量："<<lines.size()<<"("<<Config::lsdNFeatures()<<")"<<endl;
            // compute lbd descriptor
            lbd->compute( EDlines_img, lines, ldesc);
        }
    }
}


void StereoFrame::fuseStereoLines(Mat img,vector<KeyLine> & lines_)
{
    //************************** 将提取到的线特征进行融合 *************************
            //将断裂的线融合为一条完整的线
            for(vector<KeyLine>::iterator it=lines_.begin();it!=lines_.end();it++)
            {
                KeyLine src_l=*it; //取出一条线段
                //遍历所有线段，选出候选融合线段
                for(vector<KeyLine>::iterator itl=lines_.begin();itl!=lines_.end();itl++)
                {
                    if(src_l.class_id==itl->class_id || itl->class_id == -1)
                    {
                    continue; //如果是直线本身,或者该线段被删除，跳过
                    }
                    
                    //角度阈值滤波
                    double angle_d=abs(src_l.angle-itl->angle); //计算两直线的方向差的绝对值
                    if(angle_d < Config::useED_min_angle())
                    {
                        //水平距离和垂直距离滤波
                        if(abs(src_l.startPointX-itl->startPointX) > Config::useED_min_dis()*src_l.lineLength || 
                            abs(src_l.startPointX-itl->endPointX) > Config::useED_min_dis()*src_l.lineLength || 
                            abs(src_l.endPointX-itl->startPointX) > Config::useED_min_dis()*src_l.lineLength || 
                            abs(src_l.endPointX-itl->endPointX) > Config::useED_min_dis()*src_l.lineLength)
                        {
                            continue;  //如果两直线在水平方向相隔太远，则跳过
                        }
                        
                        if(abs(src_l.startPointY-itl->startPointY) > Config::useED_min_dis()*src_l.lineLength ||
                            abs(src_l.startPointY-itl->endPointY) > Config::useED_min_dis()*src_l.lineLength || 
                            abs(src_l.endPointY-itl->startPointY) > Config::useED_min_dis()*src_l.lineLength || 
                            abs(src_l.endPointY-itl->endPointY) > Config::useED_min_dis()*src_l.lineLength)
                        {
                            continue;  //如果两直线在垂直方向相隔太远，则跳过
                            
                        }
                        
                        
                        //计算两直线端点的距离
                        double d1=(double) sqrt( pow(src_l.startPointX - itl->endPointX, 2 ) + pow( src_l.startPointY - itl->endPointY, 2 ));//两线段起始点到结束点间的距离
                        double d2=(double) sqrt( pow(src_l.endPointX - itl->startPointX, 2 ) + pow( src_l.endPointY - itl->startPointY, 2 ));//两线段结束点到起始点间的距离
                        
                        int type=0;
                        double lines_d=0.f; //端点距离的最小值
                        if(d1<d2)
                        {
                            type=1; //直线1的起始点到直线2的结束点的距离更小，将直线1的起始点和直线2的结束点相连
                            lines_d=d1;
                        }
                        else
                        {
                            type=2;//直线1的结束点到直线2的起始点的距离更小，将直线1的结束点和直线2的起始点相连
                            lines_d=d2;
                        }
                        
                        //端点距离滤波
                        if( lines_d<Config::useED_min_distance() )
                        {
                            //只融合首尾相接情况
                            if(type == 1)
                            {
                                //直线1的起始点到直线2的结束点的距离更小，将直线1的起始点和直线2的结束点相连
                                src_l.startPointX=itl->startPointX;
                                src_l.startPointY=itl->startPointY; //将直线2的起始点设置为直线1的起始点
                            
                                double temp=src_l.angle; //原来线段的方向
                            
                                src_l.angle = atan2( ( src_l.endPointY - src_l.startPointY ), ( src_l.endPointX - src_l.startPointX ) ); 
                            
                                temp=abs(temp-src_l.angle);
                            
                                //融合后角度滤波
                                if(temp < Config::useED_min_angle()/2)
                                {
                                    src_l.sPointInOctaveX = src_l.startPointX;
                                    src_l.sPointInOctaveY = src_l.startPointY;
                                    src_l.ePointInOctaveX = src_l.endPointX;
                                    src_l.ePointInOctaveY = src_l.endPointY;
                                    src_l.lineLength = (float) sqrt( pow( src_l.startPointX - src_l.endPointX, 2 ) + pow( src_l.startPointY - src_l.endPointY, 2 ) );
                                    src_l.octave   = 0;
                                    src_l.size     = ( src_l.endPointX - src_l.startPointX ) * ( src_l.endPointY - src_l.startPointY );
                                    src_l.pt       = Point2f( ( src_l.endPointX + src_l.startPointX ) / 2, ( src_l.endPointY + src_l.startPointY ) / 2 );

                                    src_l.response = src_l.lineLength / max( img.cols, img.rows );
                                    cv::LineIterator li( img, Point2f( src_l.startPointX, src_l.startPointY ), Point2f( src_l.endPointX, src_l.endPointY ) );
                                    src_l.numOfPixels = li.count;
                            
                                    //更新直线
                                    *it=src_l;
                                    //删除直线
                                    itl->class_id=-1;
                                }
                            }
                            
                            else if(type == 2)
                            {
                                //直线1的结束点到直线2的起始点的距离更小，将直线1的结束点和直线起始2的点相连
                                src_l.endPointX=itl->endPointX;
                                src_l.endPointY=itl->endPointY; //将直线2的结束点设置为直线1的结束点
                                
                                double temp=src_l.angle;//原来线段的方向
                            
                                src_l.angle    = atan2( ( src_l.endPointY - src_l.startPointY ), ( src_l.endPointX - src_l.startPointX ) );//融合后线段的方向
                            
                                temp=abs(temp-src_l.angle);
                            
                                //融合后角度滤波
                                if(temp < Config::useED_min_angle()/2)
                                {
                                    src_l.sPointInOctaveX = src_l.startPointX;
                                    src_l.sPointInOctaveY = src_l.startPointY;
                                    src_l.ePointInOctaveX = src_l.endPointX;
                                    src_l.ePointInOctaveY = src_l.endPointY;
                                    src_l.lineLength = (float) sqrt( pow( src_l.startPointX - src_l.endPointX, 2 ) + pow( src_l.startPointY - src_l.endPointY, 2 ) );
                                    src_l.octave   = 0;
                                    src_l.size     = ( src_l.endPointX - src_l.startPointX ) * ( src_l.endPointY - src_l.startPointY );
                                    src_l.pt       = Point2f( ( src_l.endPointX + src_l.startPointX ) / 2, ( src_l.endPointY + src_l.startPointY ) / 2 );

                                    src_l.response = src_l.lineLength / max( img.cols, img.rows );
                                    cv::LineIterator li( img, Point2f( src_l.startPointX, src_l.startPointY ), Point2f( src_l.endPointX, src_l.endPointY ) );
                                    src_l.numOfPixels = li.count;
                            
                                    //更新直线
                                    *it=src_l;
                                    //删除直线
                                    itl->class_id=-1;
                                }
                            }
                        }
                    }
                }
            }
}

//线特征匹配
//lines_l：左目线特征 lines_r：右目线特征  ldesc_l_：左目线特征描述子  ldesc_r：右目线特征描述子  initial：是否初始化的标志位
void StereoFrame::matchStereoLines( vector<KeyLine> lines_l, vector<KeyLine> lines_r, Mat &ldesc_l_, Mat ldesc_r, bool initial )
{

    // Line segments stereo matching
    // --------------------------------------------------------------------------------------------------------------------
    stereo_ls.clear();  //清除双目线特征容器
    //如果没有提取线特征或者左目线特征为空、右目线特征为空则返回
    if (!Config::hasLines() || lines_l.empty() || lines_r.empty())
        return;

    std::vector<line_2d> coords;  //线特征两端点所在网格的索引
    coords.reserve(lines_l.size());  //初始化容器
    //遍历左目图形线特征
    for (const KeyLine &kl : lines_l)
        coords.push_back(std::make_pair(std::make_pair(kl.startPointX * inv_width, kl.startPointY * inv_height),
                                        std::make_pair(kl.endPointX * inv_width, kl.endPointY * inv_height))); //保存每条线特征的起始端点和结束端点各自所在网格的索引

    //Fill in grid & directions
    //
    list<pair<int, int>> line_coords; //右目图像每个网格中
    GridStructure grid(GRID_ROWS, GRID_COLS);
    std::vector<std::pair<double, double>> directions(lines_r.size());//保存线特征的方向
    //遍历每条右目线特征
    for (int idx = 0; idx < lines_r.size(); ++idx) {
        const KeyLine &kl = lines_r[idx]; //取出右目线特征

        std::pair<double, double> &v = directions[idx]; //
        v = std::make_pair((kl.endPointX - kl.startPointX) * inv_width, (kl.endPointY - kl.startPointY) * inv_height);
        normalize(v);  //归一化

        //通过线段的起始点和结束点来绘制直线，将直线上的像素点坐标存放到line_coords中（得到该线段经过哪些网格，线特征可能穿过好几个网格）
        getLineCoords(kl.startPointX * inv_width, kl.startPointY * inv_height, kl.endPointX * inv_width, kl.endPointY * inv_height, line_coords);
        for (const std::pair<int, int> &p : line_coords)
            grid.at(p.first, p.second).push_back(idx); //保存每个网格中右目线特征的索引
    }

    GridWindow w;
    w.width = std::make_pair(Config::matchingSWs(), 0);
    w.height = std::make_pair(0, 0);

    std::vector<int> matches_12; //保存与左目线特征相匹配的右目线特征的索引
    //进行网格线特征匹配，将匹配结果存入matches_12
    //1、两线特征的方向夹角是否小于阈值
    //2、两线特征的描述子是否是彼此的最佳距离，且最佳距离和次佳距离的比值是否小于阈值
    matchGrid(coords, ldesc_l, grid, ldesc_r, directions, w, matches_12);  
//    match(ldesc_l, ldesc_r, Config::minRatio12P(), matches_12);

    // bucle around lmatches
    Mat ldesc_l_aux;
    int ls_idx = 0;
    //遍历每一对线匹配对，检查是否满足线段匹配的约束
    for (int i1 = 0; i1 < matches_12.size(); ++i1) {
        const int i2 = matches_12[i1];  //取出与左目线特征相匹配的右目线特征的索引
        if (i2 < 0) continue;  //如果索引小于0，说明未成功匹配，跳过该左目线特征

        // estimate the disparity of the endpoints
        //估计端点的差异
        Vector3d sp_l; sp_l << lines_l[i1].startPointX, lines_l[i1].startPointY, 1.0; //左目直线的起始点坐标（齐次坐标）
        Vector3d ep_l; ep_l << lines_l[i1].endPointX,   lines_l[i1].endPointY,   1.0; //左目直线的结束点坐标（齐次坐标）
        Vector3d le_l; le_l << sp_l.cross(ep_l); le_l = le_l / std::sqrt( le_l(0)*le_l(0) + le_l(1)*le_l(1) );  //得到左目线特征的归一化坐标
        Vector3d sp_r; sp_r << lines_r[i2].startPointX, lines_r[i2].startPointY, 1.0;//右目直线的起始点坐标（齐次坐标）
        Vector3d ep_r; ep_r << lines_r[i2].endPointX,   lines_r[i2].endPointY,   1.0;//右目直线的结束点坐标（齐次坐标）
        Vector3d le_r; le_r << sp_r.cross(ep_r);  //右目线特征坐标

        double overlap = lineSegmentOverlapStereo( sp_l(1), ep_l(1), sp_r(1), ep_r(1) );  //计算重叠部分长度（y方向重叠长度）

        double disp_s, disp_e;//左右图线段起始点和结束点横坐标的差（视差）
        sp_r << ( sp_r(0)*( sp_l(1) - ep_r(1) ) + ep_r(0)*( sp_r(1) - sp_l(1) ) ) / ( sp_r(1)-ep_r(1) ) , sp_l(1) ,  1.0;
        ep_r << ( sp_r(0)*( ep_l(1) - ep_r(1) ) + ep_r(0)*( sp_r(1) - ep_l(1) ) ) / ( sp_r(1)-ep_r(1) ) , ep_l(1) ,  1.0;
        filterLineSegmentDisparity( sp_l.head(2), ep_l.head(2), sp_r.head(2), ep_r.head(2), disp_s, disp_e ); //过滤掉横坐标差异太大的
        // check minimal disparity
        //判断是否满足约束
        //1、直线的起始点和结束点在空间中的位置应该在相机平面的前面，即判断直线是否合法
        //2、左目线特征起始点y坐标-结束点y坐标的值是否大于阈值（剔除水平线）
        //3、右目线特征起始点y坐标-结束点y坐标的值是否大于阈值（剔除水平线）
        //4、重叠部分长度是否大于阈值
        if( disp_s >= Config::minDisp() && disp_e >= Config::minDisp()
            && std::abs( sp_l(1)-ep_l(1) ) > Config::lineHorizTh()
            && std::abs( sp_r(1)-ep_r(1) ) > Config::lineHorizTh()
            && overlap > Config::stereoOverlapTh() )
        {
            Vector3d sP_; sP_ = cam->backProjection( sp_l(0), sp_l(1), disp_s);  //将起始点像素坐标转化为相机坐标
            Vector3d eP_; eP_ = cam->backProjection( ep_l(0), ep_l(1), disp_e);  //将结束点像素坐标转换为相机坐标
            double angle_l = lines_l[i1].angle;  //直线的方向角
            if( initial )  //是否是初始化
            {
                ldesc_l_aux.push_back( ldesc_l_.row(i1) ); //将该线特征的描述子存入容器
                stereo_ls.push_back( new LineFeature(Vector2d(sp_l(0),sp_l(1)),disp_s,sP_,
                                                     Vector2d(ep_l(0),ep_l(1)),disp_e,eP_,
                                                     le_l,angle_l,ls_idx,lines_l[i1].octave) );  //创建一个新的线特征对象存入线特征容器中
                ls_idx++; //线特征索引加一
            }
            else
            {
                ldesc_l_aux.push_back( ldesc_l_.row(i1) );
                stereo_ls.push_back( new LineFeature(Vector2d(sp_l(0),sp_l(1)),disp_s,sP_,
                                                     Vector2d(ep_l(0),ep_l(1)),disp_e,eP_,
                                                     le_l,angle_l,-1,lines_l[i1].octave) );
            }
        }
    }

    ldesc_l_aux.copyTo(ldesc_l_);
}

void StereoFrame::matchLineFeatures(BFMatcher* bfm, Mat ldesc_1, Mat ldesc_2, vector<vector<DMatch>> &lmatches_12  )
{
    bfm->knnMatch( ldesc_1, ldesc_2, lmatches_12, 2);
}

void StereoFrame::filterLineSegmentDisparity( Vector2d spl, Vector2d epl, Vector2d spr, Vector2d epr, double &disp_s, double &disp_e )
{
    disp_s = spl(0) - spr(0);  //左右目起始点横坐标的差
    disp_e = epl(0) - epr(0);  //左右目结束点横坐标的差
    // if they are too different, ignore them
    if(  min( disp_s, disp_e ) / max( disp_s, disp_e ) < Config::lsMinDispRatio() )
    {
        disp_s = -1.0;
        disp_e = -1.0;
    }
}

/* Auxiliar methods */

void StereoFrame::pointDescriptorMAD( const vector<vector<DMatch>> matches, double &nn_mad, double &nn12_mad )
{

    vector<vector<DMatch>> matches_nn, matches_12;
    matches_nn = matches;
    matches_12 = matches;

    // estimate the NN's distance standard deviation
    double nn_dist_median;
    sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist() );
    nn_mad = matches_nn[int(matches_nn.size()/2)][0].distance;
    for( int j = 0; j < matches_nn.size(); j++)
        matches_nn[j][0].distance = fabsf( matches_nn[j][0].distance - nn_dist_median );
    sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist() );
    nn_mad = 1.4826 * matches_nn[int(matches_nn.size()/2)][0].distance;

    // estimate the NN's 12 distance standard deviation
    double nn12_dist_median;
    sort( matches_12.begin(), matches_12.end(), compare_descriptor_by_NN12_ratio() );
    nn_dist_median = matches_12[int(matches_12.size()/2)][0].distance / matches_12[int(matches_12.size()/2)][1].distance;
    for( int j = 0; j < matches_12.size(); j++)
        matches_12[j][0].distance = fabsf( matches_12[j][0].distance / matches_12[j][1].distance - nn_dist_median );
    sort( matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist() );
    nn12_mad =  1.4826 * matches_12[int(matches_12.size()/2)][0].distance;

}

void StereoFrame::lineDescriptorMAD( const vector<vector<DMatch>> matches, double &nn_mad, double &nn12_mad )
{

    vector<vector<DMatch>> matches_nn, matches_12;
    matches_nn = matches;
    matches_12 = matches;

    // estimate the NN's distance standard deviation
    double nn_dist_median;
    sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist() );
    nn_mad = matches_nn[int(matches_nn.size()/2)][0].distance;
    for( int j = 0; j < matches_nn.size(); j++)
        matches_nn[j][0].distance = fabsf( matches_nn[j][0].distance - nn_dist_median );
    sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist() );
    nn_mad = 1.4826 * matches_nn[int(matches_nn.size()/2)][0].distance;

    // estimate the NN's 12 distance standard deviation
    double nn12_dist_median;
    sort( matches_12.begin(), matches_12.end(), compare_descriptor_by_NN12_dist() );
    nn12_mad = matches_12[int(matches_12.size()/2)][1].distance - matches_12[int(matches_12.size()/2)][0].distance;
    for( int j = 0; j < matches_12.size(); j++)
        matches_12[j][0].distance = fabsf( matches_12[j][1].distance - matches_12[j][0].distance - nn_dist_median );
    sort( matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist() );
    nn12_mad =  1.4826 * matches_12[int(matches_12.size()/2)][0].distance;

}

//计算匹配线段的重叠部分长度
//spl_obs：左目线段起始点y坐标  epl_obs：左目线段结束点y坐标  spl_proj：右目线段起始点y坐标  epl_proj：右目线段结束点y坐标
double StereoFrame::lineSegmentOverlapStereo( double spl_obs, double epl_obs, double spl_proj, double epl_proj  )
{

    double overlap = 1.f; //初始化

    if( fabs( epl_obs - spl_obs ) > Config::lineHorizTh() ) // normal lines (verticals included) fabs（）取绝对值 Config::lineHorizTh()避免水平线（像素）的参数
    {
        double sln    = min(spl_obs,  epl_obs);  //取出左目线段y坐标的最小值
        double eln    = max(spl_obs,  epl_obs); //取出左目线段y坐标的最大值
        double spn    = min(spl_proj, epl_proj);//取出右目线段y坐标的最小值
        double epn    = max(spl_proj, epl_proj);//取出右目线段y坐标的最大值

        double length = eln-spn;//计算左目结束点到右目起始点的距离

        if ( (epn < sln) || (spn > eln) )
            overlap = 0.f;  //如果右目结束点小于左目起始点或者右目起始点大于左目结束点，重叠部分长度为0
        else{  //有重叠部分
            if ( (epn>eln) && (spn<sln) )
                overlap = eln-sln;  //如果右目结束点大于左目结束点并且右目起始点点小于左目起始点（右目线段完全包含左目线段），重叠部分即为左目线段的长度
            else
                overlap = min(eln,epn) - max(sln,spn);  //部分重叠，则用左右目结束点的最小值-左右目起始点的最大值
        }

        if(length>0.01f)
            overlap = overlap / length;
        else
            overlap = 0.f;

        if( overlap > 1.f )
            overlap = 1.f;

    }

    return overlap;

}

double StereoFrame::lineSegmentOverlap( Vector2d spl_obs, Vector2d epl_obs, Vector2d spl_proj, Vector2d epl_proj  )
{

    double overlap = 1.f;

    if( fabs(spl_obs(0)-epl_obs(0)) < 1.0 )         // vertical lines
    {

        // line equations
        Vector2d l = epl_obs - spl_obs;

        // intersection points
        Vector2d spl_proj_line, epl_proj_line;
        spl_proj_line << spl_obs(0), spl_proj(1);
        epl_proj_line << epl_obs(0), epl_proj(1);

        // estimate overlap in function of lambdas
        double lambda_s = (spl_proj_line(1)-spl_obs(1)) / l(1);
        double lambda_e = (epl_proj_line(1)-spl_obs(1)) / l(1);

        double lambda_min = min(lambda_s,lambda_e);
        double lambda_max = max(lambda_s,lambda_e);

        if( lambda_min < 0.f && lambda_max > 1.f )
            overlap = 1.f;
        else if( lambda_max < 0.f || lambda_min > 1.f )
            overlap = 0.f;
        else if( lambda_min < 0.f )
            overlap = lambda_max;
        else if( lambda_max > 1.f )
            overlap = 1.f - lambda_min;
        else
            overlap = lambda_max - lambda_min;

    }
    else if( fabs(spl_obs(1)-epl_obs(1)) < 1.0 )    // horizontal lines (previously removed)
    {

        // line equations
        Vector2d l = epl_obs - spl_obs;

        // intersection points
        Vector2d spl_proj_line, epl_proj_line;
        spl_proj_line << spl_proj(0), spl_obs(1);
        epl_proj_line << epl_proj(0), epl_obs(1);

        // estimate overlap in function of lambdas
        double lambda_s = (spl_proj_line(0)-spl_obs(0)) / l(0);
        double lambda_e = (epl_proj_line(0)-spl_obs(0)) / l(0);

        double lambda_min = min(lambda_s,lambda_e);
        double lambda_max = max(lambda_s,lambda_e);

        if( lambda_min < 0.f && lambda_max > 1.f )
            overlap = 1.f;
        else if( lambda_max < 0.f || lambda_min > 1.f )
            overlap = 0.f;
        else if( lambda_min < 0.f )
            overlap = lambda_max;
        else if( lambda_max > 1.f )
            overlap = 1.f - lambda_min;
        else
            overlap = lambda_max - lambda_min;

    }
    else                                            // non-degenerate cases
    {

        // line equations
        Vector2d l = epl_obs - spl_obs;
        double a = spl_obs(1)-epl_obs(1);
        double b = epl_obs(0)-spl_obs(0);
        double c = spl_obs(0)*epl_obs(1) - epl_obs(0)*spl_obs(1);

        // intersection points
        Vector2d spl_proj_line, epl_proj_line;
        double lxy = 1.f / (a*a+b*b);

        spl_proj_line << ( b*( b*spl_proj(0)-a*spl_proj(1))-a*c ) * lxy,
                         ( a*(-b*spl_proj(0)+a*spl_proj(1))-b*c ) * lxy;

        epl_proj_line << ( b*( b*epl_proj(0)-a*epl_proj(1))-a*c ) * lxy,
                         ( a*(-b*epl_proj(0)+a*epl_proj(1))-b*c ) * lxy;

        // estimate overlap in function of lambdas
        double lambda_s = (spl_proj_line(0)-spl_obs(0)) / l(0);
        double lambda_e = (epl_proj_line(0)-spl_obs(0)) / l(0);

        double lambda_min = min(lambda_s,lambda_e);
        double lambda_max = max(lambda_s,lambda_e);

        if( lambda_min < 0.f && lambda_max > 1.f )
            overlap = 1.f;
        else if( lambda_max < 0.f || lambda_min > 1.f )
            overlap = 0.f;
        else if( lambda_min < 0.f )
            overlap = lambda_max;
        else if( lambda_max > 1.f )
            overlap = 1.f - lambda_min;
        else
            overlap = lambda_max - lambda_min;

    }

    return overlap;

}

//绘制双目帧
Mat StereoFrame::plotStereoFrame()
{

    // create new image to modify it
    Mat img_l_aux;
    img_l.copyTo( img_l_aux );
    if( img_l_aux.channels() == 1 )
        cvtColor(img_l_aux, img_l_aux, CV_GRAY2BGR, 3);
    else if (img_l_aux.channels() == 4)
        cvtColor(img_l_aux, img_l_aux, CV_BGRA2BGR, 3);
    else if (img_l_aux.channels() != 3)
        throw std::runtime_error(std::string("[StereoFrame->plotStereoFrame] unsupported image format: ") +
                                 std::to_string(img_l_aux.channels()));
    img_l_aux.convertTo(img_l_aux, CV_8UC3);

    // Variables
    unsigned int    r = 0, g=0, b = 0;
    Point2f         p,q;
    double          thick = 1.5;
    int             k = 0, radius  = 3;

    // plot point features
    //绘制点特征 绿色
    for( auto pt_it = stereo_pt.begin(); pt_it != stereo_pt.end(); pt_it++)
    {
        if( (*pt_it)->inlier )
        {
            r = 0;
            g = 255;
            b = 0;
            p = cv::Point( int((*pt_it)->pl(0)), int((*pt_it)->pl(1)) );
            circle( img_l_aux, p, radius, Scalar(b,g,r), thick);
        }
    }

    // plot line segment features
    //绘制线特征 红色
    for( auto ls_it = stereo_ls.begin(); ls_it != stereo_ls.end(); ls_it++)
    {
        if( (*ls_it)->inlier )
        {
            r = 255;
            g = 0;
            b = 0;
            p = cv::Point( int((*ls_it)->spl(0)), int((*ls_it)->spl(1)) );
            q = cv::Point( int((*ls_it)->epl(0)), int((*ls_it)->epl(1)) );
            line( img_l_aux, p, q, Scalar(b,g,r), thick);
        }
    }

    return img_l_aux;//返回该图像
}

/* RGB-D Functions */

void StereoFrame::extractRGBDFeatures( double llength_th, int fast_th )
{

    bool initial = (frame_idx == 0);

    // Feature detection and description
    vector<KeyPoint> points_l;
    vector<KeyLine>  lines_l;
    if(  Config::hasPoints() && Config::hasLines() )
    {
        if( Config::plInParallel() )
        {
            auto detect_p = async(launch::async, &StereoFrame::detectPointFeaturesLeft, this, img_l, ref(points_l), ref(pdesc_l), fast_th );
            auto detect_l = async(launch::async, &StereoFrame::detectLineFeatures,  this, img_l, ref(lines_l), ref(ldesc_l), llength_th );
            detect_p.wait();
            detect_l.wait();
        }
        else
        {
            detectPointFeaturesLeft( img_l, points_l, pdesc_l, fast_th );
            detectLineFeatures( img_l, lines_l, ldesc_l, llength_th );
        }
    }
    else
    {
        if( Config::hasPoints() )
            detectPointFeaturesLeft( img_l, points_l, pdesc_l, fast_th );
        else
            detectLineFeatures( img_l, lines_l, ldesc_l, llength_th );
    }

    // Points stereo matching
    if( Config::hasPoints() && !(points_l.size()==0) )
    {
        // bucle around pmatches
        stereo_pt.clear();
        int pt_idx = 0;
        Mat pdesc_l_;
        for( int i = 0; i < points_l.size(); i++ )
        {
            if( img_r.type() == CV_32FC1 )
            {
                // read the depth for each point and estimate disparity
                float depth  = img_r.at<float>(points_l[i].pt.y,points_l[i].pt.x);
                // check correct depth values
                if( depth > Config::rgbdMinDepth() && depth < Config::rgbdMaxDepth() )
                {
                    // check minimal disparity
                    double disp   = cam->getFx() * cam->getB() / depth; // TUM factor (read also if in the ICL-NUIM is different)
                    if( disp >= Config::minDisp() ){
                        pdesc_l_.push_back( pdesc_l.row(i) );
                        Vector2d pl_; pl_ << points_l[i].pt.x, points_l[i].pt.y;
                        Vector3d P_;  P_ = cam->backProjection( pl_(0), pl_(1), disp);
                        stereo_pt.push_back( new PointFeature(pl_,disp,P_,-1) );
                        pt_idx++;
                    }
                }

            }
            else if( img_r.type() == CV_16UC1 )
            {
                // read the depth for each point and estimate disparity
                ushort depth  = img_r.at<ushort>(points_l[i].pt.y,points_l[i].pt.x);
                double depthd = double(depth/5000.0) ;
                // check correct depth values
                if( depthd > Config::rgbdMinDepth() && depthd < Config::rgbdMaxDepth() )
                {
                    // check minimal disparity
                    double disp   = cam->getFx() * cam->getB() / depthd; // TUM factor (read also if in the ICL-NUIM is different)
                    if( disp >= Config::minDisp() ){
                        pdesc_l_.push_back( pdesc_l.row(i) );
                        Vector2d pl_; pl_ << points_l[i].pt.x, points_l[i].pt.y;
                        Vector3d P_;  P_ = cam->backProjection( pl_(0), pl_(1), disp);
                        stereo_pt.push_back( new PointFeature(pl_,disp,P_,-1) );
                        pt_idx++;
                    }
                }
            }
        }
        pdesc_l_.copyTo(pdesc_l);
    }

    // Line segments stereo matching
    if( Config::hasLines() && !lines_l.empty() )
    {
        stereo_ls.clear();
        Mat ldesc_l_;
        int ls_idx = 0;
        for( int i = 0; i < lines_l.size(); i++ )
        {
            if( img_r.type() == CV_32FC1 )
            {
                // read the depth for each point and estimate disparity
                float depth_s  = img_r.at<float>(lines_l[i].pt.y,lines_l[i].pt.x);
                float depth_e  = img_r.at<float>(lines_l[i].pt.y,lines_l[i].pt.x);
                // discard points with bad depth estimation
                if( depth_s > Config::rgbdMinDepth() && depth_s < Config::rgbdMaxDepth() && depth_e > Config::rgbdMinDepth() && depth_e < Config::rgbdMaxDepth() )
                {
                    // estimate the disparity of the endpoints
                    double disp_s = cam->getFx() * cam->getB() / depth_s;
                    double disp_e = cam->getFx() * cam->getB() / depth_e;
                    filterLineSegmentDisparity(disp_s,disp_e);
                    Vector3d sp_l; sp_l << lines_l[i].startPointX, lines_l[i].startPointY, 1.0;
                    Vector3d ep_l; ep_l << lines_l[i].endPointX,   lines_l[i].endPointY,   1.0;
                    Vector3d le_l; le_l << sp_l.cross(ep_l); le_l = le_l / sqrt( le_l(0)*le_l(0) + le_l(1)*le_l(1) );
                    // check minimal disparity
                    if( disp_s >= Config::minDisp() && disp_e >= Config::minDisp() )
                    {
                        ldesc_l_.push_back( ldesc_l.row(i) );
                        Vector3d sP_; sP_ = cam->backProjection( sp_l(0), sp_l(1), disp_s);
                        Vector3d eP_; eP_ = cam->backProjection( ep_l(0), ep_l(1), disp_e);
                        double angle_l = lines_l[i].angle;
                        stereo_ls.push_back( new LineFeature(Vector2d(sp_l(0),sp_l(1)),disp_s,sP_,Vector2d(ep_l(0),ep_l(1)),disp_e,eP_,le_l,angle_l,-1) );
                        ls_idx++;
                    }
                }
            }
            else //if( img_r.type() == CV_16UC1 )
            {
                // read the depth for each point and estimate disparity
                ushort depth_s  = img_r.at<ushort>(lines_l[i].pt.y,lines_l[i].pt.x);
                ushort depth_e  = img_r.at<ushort>(lines_l[i].pt.y,lines_l[i].pt.x);
                double depthd_s = double(depth_s/5000.0) ;
                double depthd_e = double(depth_e/5000.0) ;
                // discard points with bad depth estimation
                if( depthd_s > Config::rgbdMinDepth() && depthd_s < Config::rgbdMaxDepth() &&  depthd_e > Config::rgbdMinDepth() && depthd_e < Config::rgbdMaxDepth() )
                {
                    // estimate the disparity of the endpoints
                    double disp_s = cam->getFx() * cam->getB() / depthd_s;
                    double disp_e = cam->getFx() * cam->getB() / depthd_e;
                    filterLineSegmentDisparity(disp_s,disp_e);
                    Vector3d sp_l; sp_l << lines_l[i].startPointX, lines_l[i].startPointY, 1.0;
                    Vector3d ep_l; ep_l << lines_l[i].endPointX,   lines_l[i].endPointY,   1.0;
                    Vector3d le_l; le_l << sp_l.cross(ep_l); le_l = le_l / sqrt( le_l(0)*le_l(0) + le_l(1)*le_l(1) );
                    // check minimal disparity
                    if( disp_s >= Config::minDisp() && disp_e >= Config::minDisp() )
                    {
                        ldesc_l_.push_back( ldesc_l.row(i) );
                        Vector3d sP_; sP_ = cam->backProjection( sp_l(0), sp_l(1), disp_s);
                        Vector3d eP_; eP_ = cam->backProjection( ep_l(0), ep_l(1), disp_e);
                        double angle_l = lines_l[i].angle;
                        stereo_ls.push_back( new LineFeature(Vector2d(sp_l(0),sp_l(1)),disp_s,sP_,Vector2d(ep_l(0),ep_l(1)),disp_e,eP_,le_l,angle_l,-1) );
                        ls_idx++;
                    }
                }
            }

        }
        ldesc_l_.copyTo(ldesc_l);
    }

}

void StereoFrame::filterLineSegmentDisparity( double &disp_s, double &disp_e )
{

    // TODO: ask David for bresenham to filter the line with the depth along it!!!!

    // if they are too different, ignore them
    if(  min( disp_s, disp_e ) / max( disp_s, disp_e ) < Config::lsMinDispRatio() )
    {
        disp_s = -1.0;
        disp_e = -1.0;
    }
}

}
