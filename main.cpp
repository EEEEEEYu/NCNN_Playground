#include <iostream>
#include <mat.h>
#include <layer.h>

using namespace ncnn;
using namespace std;

void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt)
{
    bottom_blob_bordered = bottom_blob;
    return;
}

int conv3d(const Mat& bottom_blob, Mat& top_blob, const Option& opt,
           const vector<int>& stride, const vector<int>& kernel, const vector<int>& dilation) {
    // test parameters
    int stride_w = stride[0];
    int stride_h = stride[1];
    int stride_d = stride[2];

    int kernel_w = kernel[0];
    int kernel_h = kernel[1];
    int kernel_d = kernel[2];

    int dilation_w = dilation[0];
    int dilation_h = dilation[1];
    int dilation_d = dilation[2];

    int num_output = 17;

    int bias_term = 0;
    int* bias_data = nullptr;

    // Generate kernel with value = 1.0
    int kernel_size = kernel_w * kernel_d * kernel_h;
    float* weight_data = (float *)malloc(num_output * bottom_blob.c * kernel_size * sizeof(float));
    float temp_offset = 0.01;
    for(int i = 0; i < num_output * bottom_blob.c * kernel_size; ++i) {
        weight_data[i] = 1.0f + temp_offset;
        temp_offset += 0.01;
    }

    int activation_type = 0;
    int* activation_params = nullptr;

    /************************************** Core Logic ***************************************/

    // record shape info
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    // Compute dilated kernel size. This extension is achieved by calculating offsets on original
    // input, instead of extending kernel itself
    const int kernel_extend_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extend_h = dilation_h * (kernel_h - 1) + 1;
    const int kernel_extend_d = dilation_d * (kernel_d - 1) + 1;

    // pad bottom blob
    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    // update shape info
    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;
    d = bottom_blob_bordered.d;

    // compute output shape
    int outw = (w - kernel_extend_w) / stride_w + 1;
    int outh = (h - kernel_extend_h) / stride_h + 1;
    int outd = (d - kernel_extend_d) / stride_d + 1;

    // compute kernel size
    const int maxk = kernel_w * kernel_h * kernel_d;

    cout << "Output shape (w,h,d): " << "(" << outw << "," << outh << "," << outd << ")" << endl;
    cout << "Output kernel_size: " << maxk << endl;

    // compute offset to align original input and kernel data
    std::vector<int> _space_ofs(maxk);
    cout << "(w,h,d): " << "(" << w << "," << h << "," << d << ")" << endl;
    cout << "(di_w,di_h,di_d): " << "(" << dilation_w << "," << dilation_h << "," << dilation_d << ")" << endl;

    int offset0 = dilation_d;
    cout << "Offset0: " << offset0 << endl;
    int offset1 = d * dilation_h - kernel_d * dilation_d;
    cout << "Offset1: " << offset1 << endl;
    int offset2 = (h*d) * dilation_w - h * kernel_h * dilation_h - kernel_h * dilation_h;
    cout << "Offset2: " << offset2 << endl;
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        for(int i = 0; i < kernel_w; ++i) {
            for(int j = 0; j < kernel_h; ++j) {
                for(int k = 0; k < kernel_d; ++k) {
                    space_ofs[p1] = p2;
                    p1++;
                    p2 += offset0;
                }
                p2 += offset1;
            }
            p2 += offset2;
        }
    }

    // create top blob
    top_blob.create(outw, outh, outd, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // main loop
    // #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int j = 0; j < outw; j++)
        {
            for (int i = 0; i < outh; i++)
            {
                for (int k = 0; k < outd; k++)
                {
                    float sum = 0.f;

                    if (bias_term)
                        sum = bias_data[p];

                    const float* kptr = (const float*)weight_data + maxk * channels * p;

                    for (int q = 0; q < channels; q++)
                    {
                        const Mat m = bottom_blob_bordered.channel(q);
                        // (w*d): offset when you go across one h
                        // (d): offset when you go across one w
                        const float* sptr = (float*)m.data + (h*d) * j * stride_w + (d) * i * stride_h + k * stride_d;

                        for (int l = 0; l < maxk; l++)
                        {
                            float val = sptr[space_ofs[l]];
                            float wt = kptr[l];
                            sum += val * wt;
                        }

                        kptr += maxk;
                    }

                    if (activation_type == 1)
                    {
                        sum = std::max(sum, 0.f);
                    }
                    else if (activation_type == 2)
                    {
                        float slope = activation_params[0];
                        sum = sum > 0.f ? sum : sum * slope;
                    }
                    else if (activation_type == 3)
                    {
                        float min = activation_params[0];
                        float max = activation_params[1];
                        if (sum < min)
                            sum = min;
                        if (sum > max)
                            sum = max;
                    }
                    else if (activation_type == 4)
                    {
                        sum = static_cast<float>(1.f / (1.f + exp(-sum)));
                    }
                    else if (activation_type == 5)
                    {
                        const float MISH_THRESHOLD = 20;
                        float x = sum, y;
                        if (x > MISH_THRESHOLD)
                            y = x;
                        else if (x < -MISH_THRESHOLD)
                            y = expf(x);
                        else
                            y = logf(expf(x) + 1);
                        sum = static_cast<float>(x * tanh(y));
                    }

                    outptr[k] = sum;
                }

                // move forward output pointer
                outptr += outd;
            }
        }
    }

    return 0;

    /************************************** Core Logic ***************************************/

    delete weight_data;
}

void print_mat(const Mat& input) {
    int size = input.w * input.h * input.d;
    for(int i = 0; i < input.c; ++i) {
        cout << "Channel " << i+1 << endl;
        for(int j = 0; j < size; ++j) {
            cout << ((float*)input.data)[i*input.cstep + j] << " ";
        }
        cout << endl;
    }
}

void test_func(int c, int w, int h, int d, const vector<int>& stride, const vector<int>& kernel, const vector<int>& dilation) {
    // Generate data
    float* data = (float *)malloc(c * w * h * d * sizeof(float));
    for(int i = 1; i < c*w*h*d+1; ++i) {
        data[i-1] = static_cast<float>(i);
    }
    Mat input = Mat(4,5,6,3, data);
    Mat output = Mat();

    Option opt = Option();

    // conv operation, stride, kernel, dilation
    conv3d(input, output, opt, stride, kernel, dilation);

    print_mat(output);

    delete data;
}


int main() {
    cout << "Hello, World!" << endl;

    // test case 1
    test_func(3,4,5,6,{1,1,1}, {1,1,1}, {1,1,1});

    return 0;
}
