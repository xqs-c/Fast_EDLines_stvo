
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

#ifdef HAS_MRPT
#include <sceneRepresentation.h>
#endif

#include <stereoFrame.h>   //双目帧头文件
#include <stereoFrameHandler.h>  
#include <boost/filesystem.hpp>

#include "dataset.h"  //数据集
#include "timer.h"   //计时器

using namespace StVO;

void showHelp(); //显示帮助信息

//获取输入参数
bool getInputArgs(int argc, char **argv, std::string &dataset_name, int &frame_offset, int &frame_number, int &frame_step, std::string &config_file);

int main(int argc, char **argv)
{

    // read inputs 读取输入
    string dataset_name, config_file; //定义数据集文件名、配置文件文件名
    int frame_offset = 0, frame_number = 0, frame_step = 1;  //帧偏移量、要处理的帧数量、要跳过s-1帧的参数（默认值为1）
    if (!getInputArgs(argc, argv, dataset_name, frame_offset, frame_number, frame_step, config_file)) {
        showHelp();
        return -1; //如果读取到终端输入命令的格式非法，返回-1
    }

    if (!config_file.empty()) Config::loadFromFile(config_file); //如果文件不为空，读取该路径对应的配置文件的各项参数信息

    // read dataset root dir fron environment variable  从环境变量读取数据集根目录
    //boost::filesystem::path文件路径  getenv（）获取"DATASETS_DIR"对应的环境变量/home/xqs/PL-SLAM/stvo-pl-1.0/MH_01_easy
    boost::filesystem::path dataset_path(string( getenv("DATASETS_DIR")));  
    if (!boost::filesystem::exists(dataset_path) || !boost::filesystem::is_directory(dataset_path)) {
        cout << "Check your DATASETS_DIR environment variable" << endl;
        return -1;  //如果数据集根目录对应的文件不存在或者不是一个文件夹，则显示错误信息并返回-1
    }

    dataset_path /= dataset_name;  //数据集路径
    if (!boost::filesystem::exists(dataset_path) || !boost::filesystem::is_directory(dataset_path)) {
        cout << "Invalid dataset path" << endl;
        return -1;  //如果数据集路径对应的文件不存在或不是一个文件夹，则显示错误信息并返回-1
    }

    string dataset_dir = dataset_path.string();  //将数据集路径转换为字符串形式：/home/xqs/PL-SLAM/stvo-pl-1.0/MH_01_easy/mav0
    //利用数据集中的相机配置文件（dataset_params.yaml）创建一个针孔双目相机模型
    PinholeStereoCamera*  cam_pin = new PinholeStereoCamera((dataset_path / "dataset_params.yaml").string()); //cam_pin:针孔双目相机模型
    Dataset dataset(dataset_dir, *cam_pin, frame_offset, frame_number, frame_step);  //创建一个数据集对象
    
    string scene_cfg_name;  //显示器配置文件路径
    string datasetType; //保存运行的数据集类型
    string saveTraPath; //保存运行轨迹的文件路径
    if( (dataset_dir.find("kitti")!=std::string::npos) || (dataset_dir.find("malaga")!=std::string::npos)  )
    {
        cout<<"该数据集为kitti数据集！"<<endl;
        datasetType="kitti";
        scene_cfg_name = "./config/scene_config.ini";//显示器配置文件路径
        
        saveTraPath = "/home/xqs/PL-SLAM/stvo-pl-1.0/trajout_kitti.txt";//保存轨迹的文件路径
    }
    else
    {
        cout<<"该数据集为EuRoC数据集！"<<endl;
        datasetType="euroc";
        
        string strPathTimes="/home/xqs/PL-SLAM/stvo-pl-1.0/EuRoC/EuRoC_TimeStamps/"; //数据集时间戳文件的路径
        string TimeName="V202.txt";
        strPathTimes +=TimeName;
        dataset.getTimeStamps(strPathTimes,frame_offset, frame_number, frame_step); //保存EuRoC数据集的时间戳
        
        scene_cfg_name = "./config/scene_config_indoor.ini"; //显示器配置文件路径
        
        saveTraPath = "/home/xqs/PL-SLAM/stvo-pl-1.0/trajout_euroc.txt";//保存轨迹的文件路径
    }
    
//     int count=0;
//     for(int i=0;i<dataset.vTimeStamps.size();i++)
//     {
//         cout<<setprecision(18)<<dataset.vTimeStamps[i]<<endl;
//         count++;
//     }
//     cout<<"一共有："<<count<<" 个时间戳"<<endl;
    
    // create scene  创造场景
    Matrix4d Tcw, T_inc = Matrix4d::Identity();  //初始化变换矩阵，以及变换矩阵的逆，初始化为单位矩阵
    Vector6d cov_eig;  //
    Matrix6d cov;   //
    Tcw = Matrix4d::Identity();
    Tcw << 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1;
    
    string saveTime = "/home/xqs/PL-SLAM/stvo-pl-1.0/timecount.txt";//保存程序运行时间的文件的路径
    ofstream oft;//轨迹文件输入流对象


    #ifdef HAS_MRPT  //如果有mrpt库
    sceneRepresentation scene(scene_cfg_name);  //创建一个显示器对象，并将显示器参数配置文件的参数输入
    scene.initializeScene(Tcw, false); //初始化显示器
    #endif

    Timer timer;  //创建一个定时器对象  
    double T=0.f;//保存系统运行总时间

    // initialize and run PL-StVO  初始化并运行PL-StVO
    int frame_counter = 0;  //创建一个帧计数器
    double t1;  //开始时间
    StereoFrameHandler* StVO = new StereoFrameHandler(cam_pin);  //创建一个双目视觉里程计对象
    Mat img_l, img_r;  //左右目图像帧
    while (dataset.nextFrame(img_l, img_r))  //当数据集下一帧存在，即未处理到最后一帧
    {
        if( frame_counter == 0 ) // initialize
        { 
            StVO->initialize(img_l,img_r,0);  //如果是第一帧，则进行初始化操作
            
            oft.open(saveTraPath.c_str(), ios::trunc); //如果已经存在文件，删除重新创建
            oft << fixed;
        
            //Eigen::Matrix4d T;
            
            Eigen::Matrix4d poseTran = StVO->curr_frame->Tfw;  //取出当前帧的转移矩阵
            Eigen::Matrix3d rotation =poseTran.block(0,0,3,3); //旋转矩阵
            Vector3d tranform=poseTran.col(3).head(3);//平移矩阵  
            
            if(datasetType == "euroc")
            {
                //以TUM格式存入文件
                Eigen::Quaterniond q(rotation); //将旋转矩阵转换为四元数
            
                oft << fixed;
                oft << setprecision(9)<<dataset.vTimeStamps[frame_counter]<<" "
                    <<tranform[0]<<" "<<tranform[1]<<" "<<tranform[2]<<" "
                    <<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<endl;
                oft << flush;
                oft.close(); 
            }
            
            if(datasetType == "kitti")
            {
                //以KITTI格式存入文件
                Eigen::Matrix3d Rwc=rotation; //矩阵的逆，相机坐标转换为世界坐标
                Vector3d twc=tranform; //平移矩阵，相机坐标转换为世界坐标
            
                oft << setprecision(9)<<Rwc(0,0)<< " " << Rwc(0,1) << " " << Rwc(0,2) << " " << twc.x() << " " 
                    << Rwc(1,0) << " " << Rwc(1,1) << " " << Rwc(1,2) << " " << twc.y() << " " 
                    << Rwc(2,0) << " " << Rwc(2,1) << " " << Rwc(2,2) << " " << twc.z() << endl;
                oft << flush;
                oft.close();
            }
            
        }
        else // run
        {
            // PL-StVO
            timer.start();//计时器开始计时
            StVO->insertStereoPair( img_l, img_r, frame_counter ); //插入双目图像帧  frame_counter ：帧计数器
            StVO->optimizePose();//进行相机位姿优化
            t1 = timer.stop();//计时器停止计时
            
            T+=t1;//计算时间总和

            T_inc   = StVO->curr_frame->DT; //相机位姿
            cov     = StVO->curr_frame->DT_cov; //相机位姿的协方差矩阵
            cov_eig = StVO->curr_frame->DT_cov_eig; //

            // update scene
            #ifdef HAS_MRPT
            //输出一些信息
            //n_inliers_pt：正确匹配的点特征数量、matched_pt.size()：n_inliers_ls：正确匹配的线特征数量
            scene.setText(frame_counter,t1,StVO->n_inliers_pt,StVO->matched_pt.size(),StVO->n_inliers_ls,StVO->matched_ls.size());
            scene.setCov( cov ); //设置协方差矩阵
            scene.setPose( T_inc ); //设置相机位姿
            scene.setImage(StVO->curr_frame->plotStereoFrame()); //在mrpt窗口中绘制图像帧，plotStereoFrame()函数复制一份图像帧并在帧中绘制点线特征
            scene.updateScene(StVO->matched_pt, StVO->matched_ls);  //更新mrpt窗口
            #endif
            
            Eigen::Matrix4d poseTran = StVO->curr_frame->Tfw;  //取出当前帧的转移矩阵,相机坐标转换为世界坐标
            Eigen::Matrix3d rotation =poseTran.block(0,0,3,3); //旋转矩阵
            Vector3d tranform=poseTran.col(3).head(3);//平移矩阵  
            
            oft.open(saveTraPath.c_str(), ios::app);
            oft << fixed;
            if(datasetType == "euroc")
            {
                //以TUM格式存入文件
                Eigen::Quaterniond q(rotation); //将旋转矩阵转换为四元数
            
                oft << fixed;
                oft << setprecision(9)<<dataset.vTimeStamps[frame_counter]<<" "
                    <<tranform[0]<<" "<<tranform[1]<<" "<<tranform[2]<<" "
                    <<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<endl;
                oft << flush;
                oft.close(); 
            }
            
            if(datasetType == "kitti")
            {
                //以KITTI格式存入文件
                Eigen::Matrix3d Rwc=rotation; //矩阵的逆，相机坐标转换为世界坐标
                Vector3d twc=tranform; //平移矩阵，相机坐标转换为世界坐标
            
                oft << setprecision(9)<<Rwc(0,0)<< " " << Rwc(0,1) << " " << Rwc(0,2) << " " << twc.x() << " " 
                    << Rwc(1,0) << " " << Rwc(1,1) << " " << Rwc(1,2) << " " << twc.y() << " " 
                    << Rwc(2,0) << " " << Rwc(2,1) << " " << Rwc(2,2) << " " << twc.z() << endl;
                oft << flush;
                oft.close();
            }

            // console output 控制台输出
            cout.setf(ios::fixed,ios::floatfield); cout.precision(8);
            cout << "Frame: " << frame_counter << "\tRes.: " << StVO->curr_frame->err_norm<<" "
            <<StVO->curr_frame->prev_frame_err_norm<<" "<<StVO->curr_frame->pprev_frame_err_norm;
            cout.setf(ios::fixed,ios::floatfield); cout.precision(3);
            cout << " \t Proc. time: " << t1 << " ms\t ";
            if( Config::adaptativeFAST() )  cout << "\t FAST: "   << StVO->orb_fast_th;
            if( Config::hasPoints())        cout << "\t Points: " << StVO->matched_pt.size() << " (" << StVO->n_inliers_pt << ") " ;
            if( Config::hasLines() )        cout << "\t Lines:  " << StVO->matched_ls.size() << " (" << StVO->n_inliers_ls << ") " ;
            cout << endl;

            // update StVO
            StVO->updateFrame();
        }

        frame_counter++;
    }
    
    ofstream ofs;  //时间文件输入流对象
    ofs.open(saveTime.c_str(),ios::out | ios::app);
    ofs<<" 总用时为： "<<T<<" ms"<<endl;
    ofs.close();
    
    cout<<" 总用时为： "<<T<<" ms"<<endl;

    // wait until the scene is closed
    #ifdef HAS_MRPT
    while( scene.isOpen() );
    #endif

    return 0;
}

//显示帮助信息
void showHelp() {
    cout << endl << "Usage: ./imagesStVO <dataset_name> [options]" << endl
         << "Options:" << endl
         << "\t-c Config file" << endl  //配置文件
         << "\t-o Offset (number of frames to skip in the dataset directory" << endl  //偏移量：数据集目录中要跳过的帧数
         << "\t-n Number of frames to process the sequence" << endl   //处理序列的帧数
         << "\t-s Parameter to skip s-1 frames (default 1)" << endl    //用于跳过s-1帧的参数（默认值为1）
         << endl;
}

//
bool getInputArgs(int argc, char **argv, std::string &dataset_name, int &frame_offset, int &frame_number, int &frame_step, std::string &config_file) {

    if( argc < 2 || argc > 10 || (argc % 2) == 1 ) //如果输入参数数量非法，返回false
        return false;

    dataset_name = argv[1];  //数据集路径
    int nargs = argc/2 - 1;  //输入参数必须为偶数，然后-1得到可选选项的参数
    for( int i = 0; i < nargs; i++ ) //遍历可选选项参数
    {
        int j = 2*i + 2;  //跳过可执行程序参数和数据集路径参数，并且选择偶数项输入参数即位可选选项参数
        if( string(argv[j]) == "-o" )
            frame_offset = stoi(argv[j+1]); //stoi（字符串，起始位置，几进制)）：将n进制的字符串转化为十进制  读取偏移量参数值
        else if( string(argv[j]) == "-n" )
            frame_number = stoi(argv[j+1]); //读取要处理的帧数量
        else if( string(argv[j]) == "-s" )
            frame_step = stoi(argv[j+1]);  //读取要跳过s-1帧的参数
        else if (string(argv[j]) == "-c")
            config_file = string(argv[j+1]);  //读取配置文件的路径
        else
            return false;  //如果可选选项参数错误，返回false
    }

    return true;
}

