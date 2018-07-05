#pragma once
#include "Constants.h"
namespace Localization
{
    class Edge
    {
    private:
        float mLength = 0;
        float mOrientation = 0;

        friend class cereal::access;
        template<class Archive>
        void serialize(Archive & archive)
        {
            archive(mLength, mOrientation);
        }
    public:
        Edge() {};

        Edge(const float& length) : mLength(length) {};

        Edge(const float& length, const float& orientation) :
            mLength(length), mOrientation(orientation) {};

        ~Edge() {};

        const float& GetLength() { return mLength; }
        const float& GetOrientation() { return mOrientation; }
        void SetLength(const float& length) { mLength = length; }
        void SetOrientation(const float& orientation) { mOrientation = orientation; }
    };


}
