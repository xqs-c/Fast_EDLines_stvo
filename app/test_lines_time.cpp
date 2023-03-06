//测试LSD、FLD、EDLines算法对同一张图的提取效果以及运行时间

#include <iostream>
#include <fstream>
#include "EDLib.h" //线特征提取器头文件

#include <chrono>
#include <config.h>
#include "lineIterator.h"
#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>

#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/fast_line_detector.hpp>

#include <line_descriptor_custom.hpp>
#include <line_descriptor/descriptor_custom.hpp>
using namespace cv;
using namespace line_descriptor;
using namespace std;
using namespace StVO;


int main(int argc,char** argv)
{
    if(argc != 2){
        cout<<"usage: feature_extraction img"<<endl;
        return 1;
    }
    
    string timePath="/home/xqs/PL-SLAM/stvo-pl-1.0/build/timecount.txt";
    ofstream oft;//轨迹文件输入流对象
    oft.open(timePath.c_str(), ios::out | ios::app); //如果已经存在文件，删除重新创建
    
    //************************** 输入图片 并显示源图像 **************************
    Mat testImg = imread(argv[1], IMREAD_GRAYSCALE);	
    assert(testImg.data != nullptr);
    imshow("Source Image", testImg);
    waitKey(0);
    
    //************************** 进行EDLines线特征提取，并显示结果 **************************
    
    //EDPF testEDPF = EDPF(testImg); //无参边缘提取器
    chrono::steady_clock::time_point t1=chrono::steady_clock::now();//记录开始时间
    EDLines testEDLines = EDLines(testImg);//线特征提取
    chrono::steady_clock::time_point t2=chrono::steady_clock::now();//记录结束时间
    chrono::duration<double> time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
    std::cout<<"time EDLines extraction: "<<time_used.count()<<"s"<<std::endl;
    int noLines = testEDLines.getLinesNo();
    std::cout<<"EDLines extraction: "<<noLines<<std::endl;
    oft<<"time EDLines extraction: "<<time_used.count()<<"s"<<std::endl;
    oft<<"EDLines extraction: "<<noLines<<std::endl;
    Mat Image_EDLines = testEDLines.drawOnImage();	//draws on the input image
	imshow("EDLines Image", Image_EDLines);
    imwrite("EDLines_Image.png",Image_EDLines);
    waitKey(0);
    
    //************************** 进行LSD线特征提取，并显示结果 **************************
     Ptr<line_descriptor::LSDDetectorC> lsd = line_descriptor::LSDDetectorC::createLSDDetectorC();  //创建一个LSD线特征提取器
     vector<KeyLine> lines;
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
    unsigned int weight=testImg.cols;
    unsigned int height=testImg.rows;
    double min_line_length  = Config::minLineLength() * std::min( weight, height) ; //线段的最小长度阈值，使用相机长、宽中的较小值*阈值
    opts.min_length   = min_line_length;  //线段的最小长度
    t1=chrono::steady_clock::now();//记录开始时间
    lsd->detect( testImg, lines, Config::lsdScale(), 1, opts); //提取线特征
    t2=chrono::steady_clock::now();//记录结束时间
    time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
    std::cout<<"time LSD extraction: "<<time_used.count()<<"s"<<std::endl;
    std::cout<<" LSD extraction: "<<lines.size()<<std::endl;
    oft<<"time LSD extraction: "<<time_used.count()<<"s"<<std::endl;
    oft<<" LSD extraction: "<<lines.size()<<std::endl;
    
    Mat Image_LSD = Mat(testImg.size(), CV_8UC3);
	cvtColor(testImg, Image_LSD, COLOR_GRAY2BGR);
	for (vector<KeyLine>::iterator it =lines.begin();it!=lines.end();it++) {
        cv::Point startPoint= cv::Point(int(it->startPointX),int(it->startPointY));
		cv::Point endPoint   = cv::Point(int(it->endPointX),  int(it->endPointY));
		line(Image_LSD, startPoint, endPoint, Scalar(0, 0, 255), 1, LINE_AA, 0); // draw lines as green on image
        //line(colorImage, linePoints[i].start, linePoints[i].end, Scalar(0, 255, 0),1,LINE_AA);
	}
	imshow("LSD Lines image",Image_LSD);
    imwrite("LSD_Image.png",Image_LSD);
    waitKey(0);
    
    //************************** 进行FLD线特征提取，并显示结果 **************************
    Mat fld_img, img_gray;
    vector<Vec4f> fld_lines;

    if( testImg.channels() != 1 )
    {
        cv::cvtColor( testImg, img_gray, CV_RGB2GRAY );
        img_gray.convertTo( fld_img, CV_8UC1 );
    }
    else
        testImg.convertTo( fld_img, CV_8UC1 );

    Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector(min_line_length);
    t1=chrono::steady_clock::now();//记录开始时间
    fld->detect( fld_img, fld_lines );
    t2=chrono::steady_clock::now();//记录结束时间
    time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
    std::cout<<"time FLD extraction: "<<time_used.count()<<"s"<<std::endl;
    std::cout<<"FLD extraction: "<<fld_lines.size()<<std::endl;
    oft<<"time FLD extraction: "<<time_used.count()<<"s"<<std::endl;
    oft<<"FLD extraction: "<<fld_lines.size()<<"\n"<<std::endl;

//     // filter lines
//     if( fld_lines.size()>Config::lsdNFeatures() && Config::lsdNFeatures()!=0  )
//     {
//         // sort lines by their response
//         sort( fld_lines.begin(), fld_lines.end(), sort_flines_by_length() );
//         fld_lines.resize(Config::lsdNFeatures());
//     }

    // loop over lines object transforming into a vector<KeyLine>
    vector<KeyLine> lines_fld;
    lines_fld.reserve(fld_lines.size());
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

        lines_fld.push_back( kl );

    }
    
    Mat Image_FLD = Mat(testImg.size(), CV_8UC3);
	cvtColor(testImg, Image_FLD, COLOR_GRAY2BGR);
	for (vector<KeyLine>::iterator it =lines_fld.begin();it!=lines_fld.end();it++) {
        cv::Point startPoint= cv::Point(int(it->startPointX),int(it->startPointY));
		cv::Point endPoint   = cv::Point(int(it->endPointX),  int(it->endPointY));
		line(Image_FLD, startPoint, endPoint, Scalar(0, 0, 255), 1, LINE_AA, 0); // draw lines as green on image
        //line(colorImage, linePoints[i].start, linePoints[i].end, Scalar(0, 255, 0),1,LINE_AA);
	}
	imshow("FLD Lines image",Image_FLD);
    imwrite("FLD_Image.png",Image_FLD);
    waitKey(0);
    
    oft.close();

    return 0;
}
