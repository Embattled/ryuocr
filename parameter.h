#pragma once
#ifndef _PARAMETER.H
#define _PARAMETER.H


namespace ryu
{
    class parameter
    {
        public:

        parameter();
        parameter(char* fileName);

    private:
        int readFromFile(char* fileName);

} // namespace ryu


#endif