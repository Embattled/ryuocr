#include <iostream>
#include <cstdio>

#include "ryuimg/ryudataset.h"
#include "ryuimg/ryumat.h"


int main(int argc, char const *argv[])
{
    ryu::dataset mydataset;
    mydataset.loadIdic("/home/eugene/workspace/horie/home_link/JPSC1400.idic");
    mydataset.printInfo();
    return 0;
}
