#include<iostream>
#include<vector>
#include<random>
#include<cmath>
#include<ctime>
#include<functional>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define IMAGE_MULTIPLIER 255 //max grayscale value
#define SAFE_PROPERTY 0.98f //prevent inverted normals

#define ALL_ZERO 0
#define RANDOM 1
#define INCREASING_LINEAR 2
#define DECREASING_LINEAR 3
#define INCREASING_SIGMOID 4
#define DECREASING_SIGMOID 5

struct WaveInformation{
    int wave_count;
    int distribution[4]; //omega frequency /phi speed /Q steepness /A amplitude
    bool if_shuffle;
};

//prototypes
void DistributeAllZero(float (*)[4],const int&,const int&);
void DistributeRandom(float (*)[4],const int&,const int&);
void DistributeIncreasingLinear(float (*)[4],const int&,const int&);
void DistributeDecreasingLinear(float (*)[4],const int&,const int&);
void DistributeIncreasingSigmoid(float (*)[4],const int&,const int&);
void DistributeDecreasingSigmoid(float (*)[4],const int&,const int&);
void BuildWaveData(WaveInformation,float (*)[4]);

int main()
{
    WaveInformation wave_inform;

    //set wave distribution information
    wave_inform.wave_count = 64; //maximum wavecount + 1
    wave_inform.distribution[0] = INCREASING_SIGMOID; //omega frequency , saved into red channel
    wave_inform.distribution[1] = RANDOM; //phi speed , saved into green channel
    wave_inform.distribution[2] = DECREASING_SIGMOID; //Q steepness  , saved into blue channel
    wave_inform.distribution[3] = ALL_ZERO; ///A amplitude , saved into alpha channel
    wave_inform.if_shuffle = false;

    //2d array for img 
    float data[2*wave_inform.wave_count][4];

    //make random vectors for wave origins
    mt19937_64 engine((float)time(NULL));
    uniform_real_distribution<float> distribution(1,100);
    auto generator = bind(distribution,engine);

    vector<tuple<float,float,float>> vectors;
    tuple<float,float,float> tmp;
    vectors.resize(wave_inform.wave_count);
   
    for(int i=0; i<wave_inform.wave_count; i++){
        tmp = make_tuple(generator(),generator(),generator());
        vectors[i]=tmp;
    }

    //normalize   
    for(int i=0;i<wave_inform.wave_count;i++){
        float length = sqrtf(powf(get<0>(vectors[i]),2)+powf(get<1>(vectors[i]),2)+powf(get<2>(vectors[i]),2));
           
        data[i][0]=SAFE_PROPERTY * IMAGE_MULTIPLIER *get<0>(vectors[i])/length;
        data[i][1]=SAFE_PROPERTY * IMAGE_MULTIPLIER *get<1>(vectors[i])/length;
        data[i][2]=SAFE_PROPERTY * IMAGE_MULTIPLIER *get<2>(vectors[i])/length;
        data[i][3]=1;
    }

    BuildWaveData(wave_inform,data);

    Mat img(Size(wave_inform.wave_count,2),CV_32FC4,data);
    cvtColor(img,img,COLOR_BGRA2RGBA);
    img.convertTo(img, CV_16UC4); 

    #ifdef __linux__
        imwrite("./WaveDataTexture.bmp",img);
    #endif

    #ifdef _WIN32
        imwrite(".\\WaveDataTexture.bmp",img);
    #endif
    return 0;
}

void DistributeAllZero(float (*data)[4],const int& data_index,const int& wave_count){
    //All data to zero, for test&debug
    for(int i=wave_count; i<2 * wave_count ; i++){
        data[i][data_index] = 0.0f;
    }
}

void DistributeRandom(float (*data)[4],const int& data_index,const int& wave_count){
    mt19937_64 engine((float)time(NULL));
    uniform_real_distribution<float> Limited(0,1);
    auto LimitedGenerator = bind(Limited,engine);
    
    for(int i=wave_count; i<2 * wave_count ; i++){
        data[i][data_index] = IMAGE_MULTIPLIER * LimitedGenerator();
    }

}

void DistributeIncreasingLinear(float (*data)[4],const int& data_index,const int& wave_count){
    //Increasing Linear distribution
    for(int i=wave_count; i<2 * wave_count ; i++){
        data[i][data_index] = IMAGE_MULTIPLIER * (static_cast<float>(i-wave_count)/wave_count);
    }
}

void DistributeDecreasingLinear(float (*data)[4],const int& data_index,const int& wave_count){
    //Decreasing Linear distribution
    for(int i=wave_count; i<2 * wave_count ; i++){
        data[3*wave_count-i-1][data_index] = IMAGE_MULTIPLIER * (static_cast<float>(i-wave_count)/wave_count);
    } 
}

void DistributeIncreasingSigmoid(float (*data)[4],const int& data_index,const int& wave_count){
    
    int sigmoid_positive_range = 10; // bigger number, more extreme curve shape

    for(int i=wave_count; i<2 * wave_count ; i++){
        //Increasing sigmoid function
        data[i][data_index] = IMAGE_MULTIPLIER *  (1/(1 + exp(-1 * (((2*(i - wave_count) - wave_count) / static_cast<float>(wave_count)) * sigmoid_positive_range))));
    }
}

void DistributeDecreasingSigmoid(float (*data)[4],const int& data_index,const int& wave_count){
    int sigmoid_positive_range = 10; // bigger number, more extreme curve shape

    for(int i=wave_count; i<2 * wave_count ; i++){
        //Decreasing sigmoid function
        data[3*wave_count-i-1][data_index] = IMAGE_MULTIPLIER * (1/(1 + exp(-1 * (((2*(i - wave_count) - wave_count) / static_cast<float>(wave_count)) * sigmoid_positive_range))));
    } 
}

void BuildWaveData(WaveInformation wave_inform,float (*data)[4]){
    
    //build wavedata
    for(int i=0;i<4;i++){
        switch(wave_inform.distribution[i]){
            case ALL_ZERO:
                DistributeAllZero(data,i,wave_inform.wave_count);
                break;
            case RANDOM:
                DistributeRandom(data,i,wave_inform.wave_count);
                break;
            case INCREASING_LINEAR:
                DistributeIncreasingLinear(data,i,wave_inform.wave_count);
                break;
            case DECREASING_LINEAR:
                DistributeDecreasingLinear(data,i,wave_inform.wave_count);
                break;
            case INCREASING_SIGMOID:
                DistributeIncreasingLinear(data,i,wave_inform.wave_count);
                break;
            case DECREASING_SIGMOID:
                DistributeDecreasingSigmoid(data,i,wave_inform.wave_count);
                break;
            default:
                break;
        }
    }

    if(wave_inform.if_shuffle){
        vector<vector<float>> shuffle_inform;
        for(int i=wave_inform.wave_count; i<2 * wave_inform.wave_count; i++){
        vector<float>tmp;
        for(int j=0;j<4;j++)
            tmp.push_back(data[i][j]);
        shuffle_inform.push_back(tmp);
        }
        random_shuffle(shuffle_inform.begin(),shuffle_inform.end());
        for(int i=wave_inform.wave_count; i<2 * wave_inform.wave_count; i++){
        for(int j=0;j<4;j++)
            data[i][j]=shuffle_inform[i-wave_inform.wave_count][j];
        }
    }
}

