#pragma once
#include "Constants.h"
namespace Localization
{
    class Path
    {
    private:
        float mLength = 0;
        float mOrientation = 0;
    public:
        Path() {};

        Path(const float& length, const float& orientation) :
            mLength(length), mOrientation(orientation) {};

        ~Path() {};

        const float& GetLength() { return mLength; }
        const float& GetOrientation() { return mOrientation; }
        void SetLength(const float& length) { mLength = length; }
        void SetOrientation(const float& orientation) { mOrientation = orientation; }
    };



}
