#include <cstdio>

#include "ryumat.h"
#include "ryudef.h"
#include "ryudataset.h"

using namespace ryu;
using namespace std;

// Init
dataset::dataset(){
    num=0;
    labels=nullptr;
    mats=nullptr;
}
dataset::~dataset(){
    clear();
}

void dataset::clear(){
    DEFFREE(labels);
    DEFFREE(mats);
    num=0;
}
void dataset::printInfo(){
    printf("Number of images: %d\n",num);
    if(num>0){
        for(int i=0;i<num;i++){
        // for(int i=0;i<num&&i<20;i++){
            printf("label: %d\t",labels[i]);
            mats[i].printInfo();
        }
    }
}
void dataset::printInfoHead(){
    printf("Number of images: %d\n",num);
    if(num>0){
        for(int i=0;i<num&&i<10;i++){
            printf("label: %d\t",labels[i]);
            mats[i].printInfo();
        }
    }
}
int dataset::initialize(int num){
    if(num<=0)return -1;
    if(this->num==num)return 0;
    this->clear();

    this->num=num;

    this->mats=DEFNEW(ryu::mat,num);
    if(!this->mats){this->clear();return -1;}

    this->labels=DEFNEW(int,num);
    if(!this->labels){this->clear();return -1;}

    return 0;
}


int dataset::loadIdic(const char* filename){
    FILE *fp = fopen(filename, "rb");
    if (!fp){ return(-1); }

    if ( readIdicData(fp) ){ fclose(fp); return(-1); }
    if ( EOF==fclose(fp) ){ return(-1); }
    return(0);
}

int dataset::readIdicData(FILE* fp){
    int num;
    if ( 1!=fread(&num, sizeof(int), 1, fp) ){ return -1; }

    if(this->initialize(num)){return -1;}

    for (int i=0 ; i<num ; i++ ){
        if ( 1!=fread(&this->labels[i], sizeof(int), 1, fp) ){
            this->clear();
            return -1;
        }
        if ( (this->mats[i]).readFromIdic(fp) ) {
             this->clear(); 
             return -1;
        }
    }
    return(0);
}
