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

    weights[get_global_id(0)] = 1.0f - randFloat(&stateValue) * 0.01f;
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

    float sum = 0.0f;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int visibleC = visibleCs[address2(visiblePosition, visibleSize.x)];

                int2 offset = visiblePosition - fieldLowerBound;

                int4 wPos;
                wPos.xyz = hiddenPosition;
                wPos.w = offset.x + offset.y * diam + visibleC * diam2;

                sum += fmax(0.0f, weights[address4(wPos, hiddenSize)] - visibleActivations[address3((int3)(visiblePosition, visibleC), visibleSize.xy)]);
            }
        }

    hiddenActivations[address3(hiddenPosition, hiddenSize.xy)] += sum;
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
                    int hiddenC = hiddenCs[address2(hiddenPosition.xy, hiddenSize.x)];

                    int2 offset = visiblePosition.xy - fieldLowerBound;

                    int4 wPos;
                    wPos.xyz = (int3)(hiddenPosition, hiddenC);
                    wPos.w = offset.x + offset.y * diam + visiblePosition.z * diam2;

                    sum += weights[address4(wPos, hiddenSize)];
                    count += 1.0f;
                }
            }
        }

    visibleActivations[address3(visiblePosition, visibleSize.xy)] = sum / fmax(1.0f, count);
}

void kernel scInhibit(global const float* hiddenActivations, global int* hiddenCs, int3 hiddenSize) {
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

    int maxIndex = 0;
    float maxValue = -99999.0f;
    
    // Find max
    for (int c = 0; c < hiddenSize.z; c++) {
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

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int visibleC = visibleCs[address2(visiblePosition, visibleSize.x)];

                int2 offset = visiblePosition - fieldLowerBound;

                for (int c = 0; c < visibleSize.z; c++) {
                    int4 wPos;
                    wPos.xyz = (int3)(hiddenPosition, hiddenC);
                    wPos.w = offset.x + offset.y * diam + c * diam2;

                    int wi = address4(wPos, hiddenSize);

                    float target = (c == visibleC ? 1.0f : 0.0f);

                    float delta = target - weights[wi];
 
                    weights[wi] += alpha * delta;
                }
            }
        }
}

// ------------------------------------------- Predictor -------------------------------------------

// Initialize weights
void kernel pInitWeights(global float* weights, uint2 seed) {
    uint2 stateValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(0) * 16 + 23) * 36;

    weights[get_global_id(0)] = (randFloat(&stateValue) * 2.0f - 1.0f) * 0.01f;
}

void kernel pForward(global const int* visibleCs, global float* hiddenActivations, global const float* weights,
    int3 visibleSize, int3 hiddenSize, float2 hiddenToVisible, int radius)
{
    int3 hiddenPosition = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	        
    int2 visiblePositionCenter = project(hiddenPosition.xy, hiddenToVisible);

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    int diam = radius * 2 + 1;
    int diam2 = diam * diam;

    float sum = 0.0f;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int visibleC = visibleCs[address2(visiblePosition, visibleSize.x)];

                int2 offset = visiblePosition - fieldLowerBound;

                int4 wPos;
                wPos.xyz = hiddenPosition;
                wPos.w = offset.x + offset.y * diam + visibleC * diam2;

                sum += weights[address4(wPos, hiddenSize)];
            }
        }

    hiddenActivations[address3(hiddenPosition, hiddenSize.xy)] += sum;
}

void kernel pInhibit(global const float* hiddenActivations, global int* hiddenCs, int3 hiddenSize) {
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

    int maxIndex = 0;
    float maxValue = -99999.0f;
    
    // Find max
    for (int c = 0; c < hiddenSize.z; c++) {
        float value = hiddenActivations[address3((int3)(hiddenPosition, c), hiddenSize.xy)];

        if (value > maxValue) {
            maxValue = value;
            maxIndex = c;
        }
    }

    // Set states
    hiddenCs[address2(hiddenPosition, hiddenSize.x)] = maxIndex;
}

void kernel pLearn(global const int* visibleCs, global const float* hiddenActivations, global const int* targetCs,
    global float* weights,
    int3 visibleSize, int3 hiddenSize, float2 hiddenToVisible, int radius, float alpha)
{
    int3 hiddenPosition = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	
    float target = (hiddenPosition.z == targetCs[address2(hiddenPosition.xy, hiddenSize.x)] ? 1.0f : 0.0f);

    float delta = alpha * (target - sigmoid(hiddenActivations[address3(hiddenPosition, hiddenSize.xy)]));

    int2 visiblePositionCenter = project(hiddenPosition.xy, hiddenToVisible);

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    int diam = radius * 2 + 1;
    int diam2 = diam * diam;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int visibleC = visibleCs[address2(visiblePosition, visibleSize.x)];

                int2 offset = visiblePosition - fieldLowerBound;

                int4 wPos;
                wPos.xyz = hiddenPosition;
                wPos.w = offset.x + offset.y * diam + visibleC * diam2;

                weights[address4(wPos, hiddenSize)] += delta;
            }
        }
}

// ------------------------------------------- Actor -------------------------------------------

// Initialize weights
void kernel aInitWeights(global float* weights, uint2 seed) {
    uint2 stateValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(0) * 16 + 23) * 36;

    weights[get_global_id(0)] = (randFloat(&stateValue) * 2.0f - 1.0f) * 0.0001f;
}

void kernel aForward(global const int* visibleCs, global float* hiddenActivations, global const float* weights,
    int3 visibleSize, int3 hiddenSize, float2 hiddenToVisible, int radius)
{
    int3 hiddenPosition = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	        
    int2 visiblePositionCenter = project(hiddenPosition.xy, hiddenToVisible);

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    int diam = radius * 2 + 1;
    int diam2 = diam * diam;

    float sum = 0.0f;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int visibleC = visibleCs[address2(visiblePosition, visibleSize.x)];

                int2 offset = visiblePosition - fieldLowerBound;

                int4 wPos;
                wPos.xyz = hiddenPosition;
                wPos.w = offset.x + offset.y * diam + visibleC * diam2;

                sum += weights[address4(wPos, hiddenSize)];
            }
        }

    hiddenActivations[address3(hiddenPosition, hiddenSize.xy)] += sum;
}

void kernel aInhibit(global const float* hiddenActivations, global int* hiddenCs, int3 hiddenSize) {
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

    int maxIndex = 0;
    float maxValue = -99999.0f;
    
    // Find max
    for (int c = 0; c < hiddenSize.z; c++) {
        float value = hiddenActivations[address3((int3)(hiddenPosition, c), hiddenSize.xy)];

        if (value > maxValue) {
            maxValue = value;
            maxIndex = c;
        }
    }

    // Set states
    hiddenCs[address2(hiddenPosition, hiddenSize.x)] = maxIndex;
}

void kernel aLearn(global const int* visibleCs, global const float* hiddenActivations, global const float* hiddenActivationsPrev,
    global const int* hiddenCs, global const int* targetCs,
    global float* weights, global float* traces,
    int3 visibleSize, int3 hiddenSize, float2 hiddenToVisible, int radius,
    float alpha, float gamma, float traceDecay, float tdErrorClip, float reward)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
    int hiddenColumnIndex = address2(hiddenPosition, hiddenSize.x);

    int targetC = targetCs[hiddenColumnIndex];
    int hiddenC = hiddenCs[hiddenColumnIndex];

    float qNext = hiddenActivations[address3((int3)(hiddenPosition, hiddenC), hiddenSize.xy)];
    float qPrev = hiddenActivationsPrev[address3((int3)(hiddenPosition, targetC), hiddenSize.xy)];

    float delta = alpha * fmin(tdErrorClip, fmax(-tdErrorClip, reward + gamma * qNext - qPrev));

    int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    int diam = radius * 2 + 1;
    int diam2 = diam * diam;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int visibleC = visibleCs[address2(visiblePosition, visibleSize.x)];

                int2 offset = visiblePosition - fieldLowerBound;

                for (int hc = 0; hc < hiddenSize.z; hc++)
                    for (int vc = 0; vc < visibleSize.z; vc++) {
                        int4 wPos;
                        wPos.xyz = (int3)(hiddenPosition, hc);
                        wPos.w = offset.x + offset.y * diam + vc * diam2;

                        int wi = address4(wPos, hiddenSize);

                        if (vc == visibleC)
                            traces[wi] = (hc == targetC ? 1.0f : 0.0f);
                        else
                            traces[wi] *= traceDecay;

                        weights[wi] += delta * traces[wi];
                    }
            }
        }
}

// ------------------------------------------- Image Encoder -------------------------------------------

// Initialize weights
void kernel imInitWeights(global float* weights, uint2 seed) {
    uint2 stateValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(0) * 16 + 23) * 36;

    weights[get_global_id(0)] = (randFloat(&stateValue) * 2.0f - 1.0f) * 0.01f;
}

void kernel imForward(global const float* visibleAs, global const float* visibleActivations,
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
                    float visibleA = visibleAs[address3((int3)(visiblePosition, c), visibleSize.xy)];

                    center += visibleA;
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
                    float visibleA = visibleAs[address3((int3)(visiblePosition, c), visibleSize.xy)];

                    int4 wPos;
                    wPos.xyz = hiddenPosition;
                    wPos.w = offset.x + offset.y * diam + c * diam2;

                    sum += weights[address4(wPos, hiddenSize)] * (visibleA - center - visibleActivations[address3((int3)(visiblePosition, c), visibleSize.xy)]);
                }
            }
        }

    hiddenActivations[address3(hiddenPosition, hiddenSize.xy)] += sum;
}

void kernel imBackward(global const int* hiddenCs, global float* visibleActivations,
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
                    int hiddenC = hiddenCs[address2(hiddenPosition.xy, hiddenSize.x)];

                    int2 offset = visiblePosition.xy - fieldLowerBound;

                    int4 wPos;
                    wPos.xyz = (int3)(hiddenPosition, hiddenC);
                    wPos.w = offset.x + offset.y * diam + visiblePosition.z * diam2;

                    sum += weights[address4(wPos, hiddenSize)];
                    count += 1.0f;
                }
            }
        }

    visibleActivations[address3(visiblePosition, visibleSize.xy)] = sum / fmax(1.0f, count);
}

void kernel imInhibit(global const float* hiddenActivations, global int* hiddenCs, int3 hiddenSize) {
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

    int maxIndex = 0;
    float maxValue = -99999.0f;
    
    // Find max
    for (int c = 0; c < hiddenSize.z; c++) {
        float value = hiddenActivations[address3((int3)(hiddenPosition, c), hiddenSize.xy)];

        if (value > maxValue) {
            maxValue = value;
            maxIndex = c;
        }
    }

    // Set states
    hiddenCs[address2(hiddenPosition, hiddenSize.x)] = maxIndex;
}

void kernel imLearn(global const float* visibleAs, global const float* visibleActivations, global const int* hiddenCs,
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
                    float visibleA = visibleAs[address3((int3)(visiblePosition, c), visibleSize.xy)];

                    center += visibleA;
                    count += 1.0f;
                }
            }
        }

    center /= fmax(1.0f, count);

    float weightSum = 0.0f;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize.xy)) {
                int2 offset = visiblePosition - fieldLowerBound;

                for (int c = 0; c < visibleSize.z; c++) {
                    float visibleA = visibleAs[address3((int3)(visiblePosition, c), visibleSize.xy)];

                    int4 wPos;
                    wPos.xyz = (int3)(hiddenPosition, hiddenC);
                    wPos.w = offset.x + offset.y * diam + c * diam2;

                    int wi = address4(wPos, hiddenSize);

                    float delta = visibleA - center - visibleActivations[address3((int3)(visiblePosition, c), visibleSize.xy)];

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
                    float visibleA = visibleAs[address3((int3)(visiblePosition, c), visibleSize.xy)];

                    int4 wPos;
                    wPos.xyz = (int3)(hiddenPosition, hiddenC);
                    wPos.w = offset.x + offset.y * diam + c * diam2;

                    int wi = address4(wPos, hiddenSize);

                    float delta = visibleA - center - visibleActivations[address3((int3)(visiblePosition, c), visibleSize.xy)];

                    float w = weights[wi] + alpha * delta;

                    weights[wi] = w * scale;
                }
            }
        }
}