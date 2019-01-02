// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// ------------------------------------------- Common -------------------------------------------

float randFloat(uint2* state) {
    const float invMaxInt = 1.0f / 4294967296.0f;
    uint x = (*state).x * 17 + (*state).y * 13123;
    (*state).x = (x << 13) ^ x;
    (*state).y ^= (x << 7);

    uint tmp = x * (x * x * 15731 + 74323) + 871483;

    return convert_float(tmp) * invMaxInt;
}

float randNormal(uint2* state) {
    float u1 = randFloat(state);
    float u2 = randFloat(state);

    return sqrt(-2.0f * log(u1)) * cos(6.28318f * u2);
}

bool inBounds0(int2 position, int2 upperBound) {
    return position.x >= 0 && position.x < upperBound.x && position.y >= 0 && position.y < upperBound.y;
}

bool inBounds(int2 position, int2 lowerBound, int2 upperBound) {
    return position.x >= lowerBound.x && position.x < upperBound.x && position.y >= lowerBound.y && position.y < upperBound.y;
}

int2 project(int2 position, float2 toScalars) {
    return (int2)(position.x * toScalars.x + 0.5f, position.y * toScalars.y + 0.5f);
}

int2 projectf(float2 position, float2 toScalars) {
    return (int2)(position.x * toScalars.x + 0.5f, position.y * toScalars.y + 0.5f);
}

int address2(int2 pos, int dim) {
    return pos.x + pos.y * dim;
}

int address3(int3 pos, int2 dims) {
    return pos.x + pos.y * dims.x + pos.z * dims.x * dims.y;
}

int address4(int4 pos, int3 dims) {
    int dxy = dims.x * dims.y;
    int dxyz = dxy * dims.z;

    return pos.x + pos.y * dims.x + pos.z * dxy + pos.w * dxyz;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// ------------------------------------------- Sparse Coder -------------------------------------------

// Initialize weights
void kernel scInitWeights(global float* weights, uint2 seed) {
    uint2 stateValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(0) * 16 + 23) * 36;

    weights[get_global_id(0)] = randFloat(&stateValue);
}

void kernel scForward(global const int* visibleCs, global const float* visibleActivations,
    global float* hiddenActivations,
    global const float* weights,
    int3 visibleSize, int3 hiddenSize, float2 hiddenToVisible, int radius)
{
    int3 hiddenPosition = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

    int2 visiblePositionCenter = project(hiddenPosition.xy, hiddenToVisible);

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    int diam = radius * 2 + 1;
    int diam2 = diam * diam;

    int4 wPos;
    wPos.xyz = hiddenPosition;

    float sum = 0.0f;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int visibleIndex = address2(visiblePosition, visibleSize.x);

                int visibleC = visibleCs[visibleIndex];

                int2 offset = visiblePosition - fieldLowerBound;

                wPos.w = offset.x + offset.y * diam + visibleC * diam2;

                sum += weights[address4(wPos, hiddenSize)] * (1.0f - visibleActivations[visibleIndex]);
            }
        }

    hiddenActivations[address3(hiddenPosition, hiddenSize.xy)] += sum;
}

void kernel scBackwardPartial(global const int* visibleCs, global const int* hiddenCs, global float* visibleActivations,
    global const float* weights,
    int3 visibleSize, int3 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int radius, int2 reverseRadii)
{
    int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));

    int visibleIndex = address2(visiblePosition, visibleSize.x);

    int visibleC = visibleCs[visibleIndex];

    int2 hiddenPositionCenter = project(visiblePosition, visibleToHidden);

    int diam = radius * 2 + 1;
    int diam2 = diam * diam;

    float sum = 0.0f;
    float count = 0.0f;

    for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++)
        for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
            int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);

            if (inBounds0(hiddenPosition, hiddenSize.xy)) {
                // Next layer node's receptive field
                int2 visibleFieldCenter = project(hiddenPosition, hiddenToVisible);

                int2 fieldLowerBound = visibleFieldCenter - (int2)(radius);
                int2 fieldUpperBound = visibleFieldCenter + (int2)(radius + 1); // So is included in inBounds

                // Check for containment
                if (inBounds(visiblePosition, fieldLowerBound, fieldUpperBound)) {
                    int hiddenC = hiddenCs[address2(hiddenPosition, hiddenSize.x)];

                    int2 offset = visiblePosition - fieldLowerBound;

                    int4 wPos;
                    wPos.xyz = (int3)(hiddenPosition, hiddenC);
                    wPos.w = offset.x + offset.y * diam + visibleC * diam2;

                    sum += weights[address4(wPos, hiddenSize)];
                    count += 1.0f;
                }
            }
        }

    visibleActivations[visibleIndex] = sigmoid(sum / fmax(1.0f, count));
}

void kernel scBackward(global const int* hiddenCs, global float* visibleActivations,
    global const float* weights,
    int3 visibleSize, int3 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int radius, int2 reverseRadii)
{
    int3 visiblePosition = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

    int2 hiddenPositionCenter = project(visiblePosition.xy, visibleToHidden);

    int diam = radius * 2 + 1;
    int diam2 = diam * diam;

    float sum = 0.0f;
    float count = 0.0f;

    for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++)
        for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
            int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);

            if (inBounds0(hiddenPosition, hiddenSize.xy)) {
                // Next layer node's receptive field
                int2 visibleFieldCenter = project(hiddenPosition, hiddenToVisible);

                int2 fieldLowerBound = visibleFieldCenter - (int2)(radius);
                int2 fieldUpperBound = visibleFieldCenter + (int2)(radius + 1); // So is included in inBounds

                // Check for containment
                if (inBounds(visiblePosition.xy, fieldLowerBound, fieldUpperBound)) {
                    int hiddenC = hiddenCs[address2(hiddenPosition, hiddenSize.x)];

                    int2 offset = visiblePosition.xy - fieldLowerBound;

                    int4 wPos;
                    wPos.xyz = (int3)(hiddenPosition, hiddenC);
                    wPos.w = offset.x + offset.y * diam + visiblePosition.z * diam2;

                    sum += weights[address4(wPos, hiddenSize)];
                    count += 1.0f;
                }
            }
        }

    visibleActivations[address3(visiblePosition, visibleSize.xy)] = sigmoid(sum / fmax(1.0f, count));
}

void kernel scInhibit(global const float* hiddenActivations, global int* hiddenCs, int3 hiddenSize) {
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

    int maxIndex = 0;
    float maxValue = hiddenActivations[address3((int3)(hiddenPosition, 0), hiddenSize.xy)];
    
    // Find max
    for (int c = 1; c < hiddenSize.z; c++) {
        float value = hiddenActivations[address3((int3)(hiddenPosition, c), hiddenSize.xy)];

        if (value > maxValue) {
            maxValue = value;
            maxIndex = c;
        }
    }

    // Set states
    hiddenCs[address2(hiddenPosition, hiddenSize.x)] = maxIndex;
}

void kernel scLearn(global const int* visibleCs, global const float* visibleActivations, global const int* hiddenCs,
    global float* weights,
    int3 visibleSize, int3 hiddenSize, float2 hiddenToVisible, int radius, float alpha)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

    int hiddenC = hiddenCs[address2(hiddenPosition, hiddenSize.x)];

    int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    int diam = radius * 2 + 1;
    int diam2 = diam * diam;

    int4 wPos;
    wPos.xyz = (int3)(hiddenPosition, hiddenC);

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int visibleC = visibleCs[address2(visiblePosition, visibleSize.x)];

                int2 offset = visiblePosition - fieldLowerBound;

                for (int c = 0; c < visibleSize.z; c++) {
                    wPos.w = offset.x + offset.y * diam + c * diam2;

                    int wi = address4(wPos, hiddenSize);

                    float target = (c == visibleC ? 1.0f : 0.0f);

                    float delta = target - visibleActivations[address3((int3)(visiblePosition, c), visibleSize.xy)];
 
                    weights[wi] += alpha * delta;
                }
            }
        }
}

// ------------------------------------------- Actor -------------------------------------------

// Initialize weights
void kernel aInitWeights(global float* weights, uint2 seed) {
    uint2 stateValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(0) * 16 + 23) * 36;

    weights[get_global_id(0)] = (randFloat(&stateValue) * 2.0f - 1.0f) * 0.001f;
}

void kernel aForward(global const int* visibleCs, global float* hiddenValues, global float* hiddenActivations,
    global const float* valueWeights, global const float* actionWeights,
    int3 visibleSize, int3 hiddenSize, float2 hiddenToVisible, int radius)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	        
    int2 visiblePositionCenter = project(hiddenPosition.xy, hiddenToVisible);

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    int diam = radius * 2 + 1;
    int diam2 = diam * diam;

    int3 wPosValue;
    wPosValue.xy = hiddenPosition;

    float value = 0.0f;
    float count = 0.0f;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int visibleC = visibleCs[address2(visiblePosition, visibleSize.x)];

                int2 offset = visiblePosition - fieldLowerBound;

                wPosValue.z = offset.x + offset.y * diam + visibleC * diam2;

                value += valueWeights[address3(wPosValue, hiddenSize.xy)];
                count += 1.0f;
            }
        }

    hiddenValues[address2(hiddenPosition, hiddenSize.x)] += value / fmax(1.0f, count);

    for (int c = 0; c < hiddenSize.z; c++) {
        int4 wPosAction;
        wPosAction.xyz = (int3)(hiddenPosition.xy, c);

        float sum = 0.0f;

        for (int dx = -radius; dx <= radius; dx++)
            for (int dy = -radius; dy <= radius; dy++) {
                int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

                if (inBounds0(visiblePosition, visibleSize.xy)) {
                    int visibleC = visibleCs[address2(visiblePosition, visibleSize.x)];

                    int2 offset = visiblePosition - fieldLowerBound;

                    wPosAction.w = offset.x + offset.y * diam + visibleC * diam2;

                    sum += actionWeights[address4(wPosAction, hiddenSize)];
                }
            }

        hiddenActivations[address3((int3)(hiddenPosition.xy, c), hiddenSize.xy)] += sum / fmax(1.0f, count);
    }
}

void kernel aInhibit(global const float* hiddenActivations, global int* hiddenCs, int3 hiddenSize, float epsilon, uint2 seed) {
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

    uint2 stateValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(0) * 16 + 23) * 36;

    int selectIndex = 0;

    if (randFloat(&stateValue) < epsilon)
        selectIndex = (int)(randFloat(&stateValue) * (hiddenSize.z - 1) + 0.5f);
    else {
        float maxValue = hiddenActivations[address3((int3)(hiddenPosition, 0), hiddenSize.xy)];
    
        // Find max
        for (int c = 1; c < hiddenSize.z; c++) {
            float value = hiddenActivations[address3((int3)(hiddenPosition, c), hiddenSize.xy)];

            if (value > maxValue) {
                maxValue = value;
                selectIndex = c;
            }
        }
    }

    // Set states
    hiddenCs[address2(hiddenPosition, hiddenSize.x)] = selectIndex;
}

void kernel aLearn(global const int* visibleCsPrev,
    global const float* hiddenValues, global const float* hiddenValuesPrev, global const float* hiddenValuesPrevPrev,
    global const float* hiddenActivationsPrev,
    global const int* hiddenCsPrev,
    global float* valueWeights, global float* actionWeights,
    int3 visibleSize, int3 hiddenSize, float2 hiddenToVisible, int radius,
    float alpha, float beta, float gamma, float qTarget)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
    int hiddenColumnIndex = address2(hiddenPosition, hiddenSize.x);

    int hiddenCPrev = hiddenCsPrev[hiddenColumnIndex];

    float qUpdate = qTarget + gamma * hiddenValues[hiddenColumnIndex];

    float deltaValue = alpha * (qUpdate - hiddenValuesPrev[hiddenColumnIndex]);
    float deltaAction = beta * (qUpdate - hiddenValuesPrevPrev[hiddenColumnIndex]);
    
    int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    int diam = radius * 2 + 1;
    int diam2 = diam * diam;

    int3 wPosValue;
    wPosValue.xy = hiddenPosition;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int visibleCPrev = visibleCsPrev[address2(visiblePosition, visibleSize.x)];

                int2 offset = visiblePosition - fieldLowerBound;

                wPosValue.z = offset.x + offset.y * diam + visibleCPrev * diam2;

                valueWeights[address3(wPosValue, hiddenSize.xy)] += deltaValue;
            }
        }
    
    for (int c = 0; c < hiddenSize.z; c++) {
        int4 wPosAction;
        wPosAction.xyz = (int3)(hiddenPosition, c);

        float delta = deltaAction * ((c == hiddenCPrev ? 1.0f : 0.0f) - sigmoid(hiddenActivationsPrev[address3((int3)(hiddenPosition, c), hiddenSize.xy)]));

        for (int dx = -radius; dx <= radius; dx++)
            for (int dy = -radius; dy <= radius; dy++) {
                int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

                if (inBounds0(visiblePosition, visibleSize.xy)) {
                    int visibleCPrev = visibleCsPrev[address2(visiblePosition, visibleSize.x)];

                    int2 offset = visiblePosition - fieldLowerBound;

                    wPosAction.w = offset.x + offset.y * diam + visibleCPrev * diam2;

                    actionWeights[address4(wPosAction, hiddenSize)] += delta;
                }
            }
    }

    // int4 wPosAction;
    // wPosAction.xyz = (int3)(hiddenPosition, hiddenCPrev);

    // for (int dx = -radius; dx <= radius; dx++)
    //     for (int dy = -radius; dy <= radius; dy++) {
    //         int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

    //         if (inBounds0(visiblePosition, visibleSize.xy)) {
    //             int visibleCPrev = visibleCsPrev[address2(visiblePosition, visibleSize.x)];

    //             int2 offset = visiblePosition - fieldLowerBound;

    //             wPosAction.w = offset.x + offset.y * diam + visibleCPrev * diam2;

    //             actionWeights[address4(wPosAction, hiddenSize)] += deltaAction;
    //         }
    //     }
}

// ------------------------------------------- Image Encoder -------------------------------------------

// Initialize weights
void kernel imInitWeights(global float* weights, uint2 seed) {
    uint2 stateValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(0) * 16 + 23) * 36;

    weights[get_global_id(0)] = randFloat(&stateValue) * 2.0f - 1.0f;
}

void kernel imForward(global const float* visibleAs, global const float* visibleAsPrev,
    global float* hiddenActivations,
    global const float* weights,
    int3 visibleSize, int3 hiddenSize, float2 hiddenToVisible, int radius)
{
    int3 hiddenPosition = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

    int2 visiblePositionCenter = project(hiddenPosition.xy, hiddenToVisible);

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    int diam = radius * 2 + 1;
    int diam2 = diam * diam;

    float center = 0.0f;
    float count = 0.0f;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int2 offset = visiblePosition - fieldLowerBound;

                for (int c = 0; c < visibleSize.z; c++) {
                    int visibleIndex = address3((int3)(visiblePosition, c), visibleSize.xy);

                    float act = visibleAs[visibleIndex] - visibleAsPrev[visibleIndex];

                    center += act;
                    count += 1.0f;
                }
            }
        }

    center /= fmax(1.0f, count);

    float sum = 0.0f;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int2 offset = visiblePosition - fieldLowerBound;

                for (int c = 0; c < visibleSize.z; c++) {
                    int visibleIndex = address3((int3)(visiblePosition, c), visibleSize.xy);

                    float act = visibleAs[visibleIndex] - visibleAsPrev[visibleIndex];

                    int4 wPos;
                    wPos.xyz = hiddenPosition;
                    wPos.w = offset.x + offset.y * diam + c * diam2;

                    float d = act - center;

                    sum += weights[address4(wPos, hiddenSize)] * d;
                }
            }
        }

    hiddenActivations[address3(hiddenPosition, hiddenSize.xy)] += sum;
}

void kernel imInhibit(global const float* hiddenActivations, global int* hiddenCs, int3 hiddenSize) {
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

    int maxIndex = 0;
    float maxValue = hiddenActivations[address3((int3)(hiddenPosition, 0), hiddenSize.xy)];
    
    // Find max
    for (int c = 1; c < hiddenSize.z; c++) {
        float value = hiddenActivations[address3((int3)(hiddenPosition, c), hiddenSize.xy)];

        if (value > maxValue) {
            maxValue = value;
            maxIndex = c;
        }
    }

    // Set states
    hiddenCs[address2(hiddenPosition, hiddenSize.x)] = maxIndex;
}

void kernel imLearn(global const float* visibleAs, global const float* visibleAsPrev, global const int* hiddenCs,
    global float* weights,
    int3 visibleSize, int3 hiddenSize, float2 hiddenToVisible, int radius, float alpha)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

    int hiddenC = hiddenCs[address2(hiddenPosition, hiddenSize.x)];

    int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    int diam = radius * 2 + 1;
    int diam2 = diam * diam;

    float center = 0.0f;
    float count = 0.0f;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int2 offset = visiblePosition - fieldLowerBound;

                for (int c = 0; c < visibleSize.z; c++) {
                    int visibleIndex = address3((int3)(visiblePosition, c), visibleSize.xy);

                    float act = visibleAs[visibleIndex] - visibleAsPrev[visibleIndex];

                    center += act;
                    count += 1.0f;
                }
            }
        }

    center /= fmax(1.0f, count);

    int4 wPos;
    wPos.xyz = (int3)(hiddenPosition, hiddenC);

    float weightSum = 0.0f;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int2 offset = visiblePosition - fieldLowerBound;

                for (int c = 0; c < visibleSize.z; c++) {
                    int visibleIndex = address3((int3)(visiblePosition, c), visibleSize.xy);

                    float act = visibleAs[visibleIndex] - visibleAsPrev[visibleIndex];

                    wPos.w = offset.x + offset.y * diam + c * diam2;

                    int wi = address4(wPos, hiddenSize);

                    float delta = act - center - weights[wi];

                    float w = weights[wi] + alpha * delta;

                    weightSum += w * w;
                }
            }
        }

    float scale = 1.0f / fmax(0.0001f, sqrt(weightSum));

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int2 offset = visiblePosition - fieldLowerBound;

                for (int c = 0; c < visibleSize.z; c++) {
                    int visibleIndex = address3((int3)(visiblePosition, c), visibleSize.xy);

                    float act = visibleAs[visibleIndex] - visibleAsPrev[visibleIndex];

                    wPos.w = offset.x + offset.y * diam + c * diam2;

                    int wi = address4(wPos, hiddenSize);

                    float delta = act - center - weights[wi];

                    float w = weights[wi] + alpha * delta;

                    weights[wi] = w * scale;
                }
            }
        }
}