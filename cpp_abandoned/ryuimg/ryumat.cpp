#include <cstdio>

#include "ryumat.h"
#include "ryudef.h"

using namespace ryu;
using namespace std;
mat::mat(){
    this->width=0;
    this->height=0;
    this->channel=0;
    this->data=nullptr;
    this->linesize=0;
    this->size=0;
}
mat::mat(int width, int height, int channel){
    this->width=width;
    this->height=height;
    this->channel=channel;
    this->data=nullptr;
}
void mat::clear(){
    DEFFREE(this->data);
    this->data=nullptr;
}
mat::~mat(){
    clear();
}

// Allocate memory
int mat::initialize()
{
    if(channel!=1&&channel!=3)return -1;
    if(width<=0||height<=0)return -1;

    this->clear();
    this->data=DEFNEW(unsigned char,width*height*channel);
    if(data==0)return -1;

    this->linesize=width*channel;
    this->size=height*linesize;
    return 0;
}
int mat::initialize(int width, int height, int channel)
{
    if(channel!=1&&channel!=3)return -1;
    if(width<=0||height<=0)return -1;
    
    this->width=width;
    this->height=height;
    this->channel=channel;
    return this->initialize();
}

int mat::getSum(){
    if(this->data==nullptr)return -1;
    int sum=0;
    for(int i=0;i<(this->size);i++){sum+=(this->data)[i];}
    return sum;
}

int mat::subAll(unsigned char x){
    if(this->data==nullptr)return -1;
    for(int i=0;i<(this->size);i++){
        this->data[i]-=x;
    }
    return 0;
}


bool mat::operator==(const mat&mat2){
    return (this->width==mat2.width)&&(this->height==mat2.height)&&(this->channel==mat2.channel)&&(this->data==mat2.data);
}
void mat::operator=(const mat&mat2){
    if(*this==mat2)return;
    if(this->data!=nullptr){
        this->clear();
    }
    this->width=mat2.width;
    this->height=mat2.height;
    this->channel=mat2.channel;
    this->data=mat2.data;
}


// io functions
int mat::readFromIdic(std::FILE* fp){
    int h, w, ch;
    if ( 1!=fread(&h, sizeof(int), 1, fp) ){ return(-1); }
    if ( 1!=fread(&w, sizeof(int), 1, fp) ){ return(-1); }
    if ( 1!=fread(&ch, sizeof(int), 1, fp) ){ return(-1); }
    if(this->initialize(w,h,ch)){
        this->clear();
        return -1;
    }
    if ( (unsigned int)this->size!=fread(this->data, sizeof(unsigned char), this->size, fp) ){
        this->clear();
        return(-1); 
    }
    return(0);
}

void mat::printInfo(){
    if(!data)
        printf("Empty mat.\n");
    else{
        printf( "Image size: %d * %d\t",width,height);
        printf("channel: %d\n",channel); 
    }
}

