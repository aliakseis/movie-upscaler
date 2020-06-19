// based on https://ffmpeg.org/doxygen/trunk/remuxing_8c-example.html

#include "TransformVideo.h"

#include <opencv2/core/utility.hpp>
//#include <opencv2/dnn_superres.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>


#include <functional>
#include <iostream>


namespace {

/** @brief Class for importing DepthToSpace layer from the ESPCN model
*/
class DepthToSpace CV_FINAL : public cv::dnn::Layer
{
public:
    DepthToSpace(const cv::dnn::LayerParams &params);

    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params);

    virtual bool getMemoryShapes(const std::vector<std::vector<int> > &inputs,
                                 const int,
                                 std::vector<std::vector<int> > &outputs,
                                 std::vector<std::vector<int> > &) const CV_OVERRIDE;

    virtual void forward(cv::InputArrayOfArrays inputs_arr,
                         cv::OutputArrayOfArrays outputs_arr,
                         cv::OutputArrayOfArrays) CV_OVERRIDE;

    /// Register this layer
    static void registerLayer()
    {
        static bool initialized = false;
        if (!initialized)
        {
            //Register custom layer that implements pixel shuffling
            std::string name = "DepthToSpace";
            cv::dnn::LayerParams layerParams = cv::dnn::LayerParams();
            cv::dnn::LayerFactory::registerLayer("DepthToSpace", DepthToSpace::create);
            initialized = true;
        }
    }
};

DepthToSpace::DepthToSpace(const cv::dnn::LayerParams &params) : Layer(params)
{
}

cv::Ptr<cv::dnn::Layer> DepthToSpace::create(cv::dnn::LayerParams &params)
{
    return cv::Ptr<cv::dnn::Layer>(new DepthToSpace(params));
}

bool DepthToSpace::getMemoryShapes(const std::vector <std::vector<int>> &inputs,
    const int, std::vector <std::vector<int>> &outputs, std::vector <std::vector<int>> &) const
{
    std::vector<int> outShape(4);

    int scale;
    if (inputs[0][1] == 4 || inputs[0][1] == 9 || inputs[0][1] == 16) //Only one image channel
    {
        scale = static_cast<int>(sqrt(inputs[0][1]));
    }
    else // Three image channels
    {
        scale = static_cast<int>(sqrt(inputs[0][1] / 3));
    }

    outShape[0] = inputs[0][0];
    outShape[1] = static_cast<int>(inputs[0][1] / pow(scale, 2));
    outShape[2] = static_cast<int>(scale * inputs[0][2]);
    outShape[3] = static_cast<int>(scale * inputs[0][3]);

    outputs.assign(4, outShape);

    return false;
}

void DepthToSpace::forward(cv::InputArrayOfArrays inputs_arr, cv::OutputArrayOfArrays outputs_arr,
    cv::OutputArrayOfArrays)
{
    std::vector <cv::Mat> inputs, outputs;
    inputs_arr.getMatVector(inputs);
    outputs_arr.getMatVector(outputs);
    cv::Mat &inp = inputs[0];
    cv::Mat &out = outputs[0];
    const float *inpData = (float *)inp.data;
    float *outData = (float *)out.data;

    const int inpHeight = inp.size[2];
    const int inpWidth = inp.size[3];

    const int numChannels = out.size[1];
    const int outHeight = out.size[2];
    const int outWidth = out.size[3];

    int scale = int(outHeight / inpHeight);
    int count = 0;

    for (int ch = 0; ch < numChannels; ch++)
    {
        for (int y = 0; y < outHeight; y++)
        {
            for (int x = 0; x < outWidth; x++)
            {
                int x_coord = static_cast<int>(floor((y / scale)));
                int y_coord = static_cast<int>(floor((x / scale)));
                int c_coord = numChannels * scale * (y % scale) + numChannels * (x % scale) + ch;

                int index = (((c_coord * inpHeight) + x_coord) * inpWidth) + y_coord;

                outData[count++] = inpData[index];
            }
        }
    }
}


} // namespace



const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{@input         |      | input file           }"
    "{@output        |      | output file          }"
//    "{m model      | fsrcnn | desired model name   }"
    "{f file | FSRCNN_x2.pb | desired model file   }"
//    "{u upscale      | 2    | upscale factor       }"
;

int main(int argc, char **argv)
{    
    try {

        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Movie Upscaler Application");
        if (parser.has("help"))
        {
            parser.printMessage();
            return 0;
        }

        const auto input = parser.get<cv::String>(0);
        const auto output = parser.get<cv::String>(1);

        //const auto modelName = parser.get<cv::String>("m");
        const auto modelFile = parser.get<cv::String>("f");
        enum { upscale = 2 };//parser.get<int>("u");

        DepthToSpace::registerLayer();

        auto net = cv::dnn::readNetFromTensorflow(modelFile);

        auto lam = [&net](cv::Mat& img) {
            //Upscale

            //Split the image: only the Y channel is used for inference
            cv::Mat ycbcr_channels[2];
            split(img, ycbcr_channels);

            auto cbcr_channel = ycbcr_channels[1].reshape(2);
            //cv::Mat cbcr_channels[2];
            std::vector <cv::Mat> cbcr_channels;
            split(cbcr_channel, cbcr_channels);

            cv::Mat Y;
            ycbcr_channels[0].convertTo(Y, CV_32F, 1.0 / 255.0);

            //Create blob from image so it has size 1,1,Width,Height
            cv::Mat blob;
            cv::dnn::blobFromImage(Y, blob, 1.0);

            //Get the HR output
            net.setInput(blob);

            cv::Mat blob_output = net.forward();

            //Convert from blob
            std::vector <cv::Mat> model_outs;
            cv::dnn::imagesFromBlob(blob_output, model_outs);
            cv::Mat out_img = model_outs[0];
            cv::Mat u_img;
            out_img.convertTo(u_img, CV_8U, 255.0);

            std::vector <cv::Mat> out_cbcr_channels(2);
            cv::resize(cbcr_channels[0], out_cbcr_channels[0], cv::Size(), upscale, upscale);
            cv::resize(cbcr_channels[1], out_cbcr_channels[1], cv::Size(), upscale, upscale);

            cv::Mat CrCb;
            merge(out_cbcr_channels, CrCb);
            auto CrCb_channel = CrCb.reshape(1);

            std::vector <cv::Mat> channels;
            channels.push_back(u_img);
            channels.push_back(CrCb_channel);

            cv::Mat merged_img;
            merge(channels, merged_img);

            img = merged_img;
        };

        return TransformVideo(input.c_str(), output.c_str(), lam, upscale);
    }
    catch (const std::exception& ex) {
        std::cerr << "Exception " << typeid(ex).name() << ": " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
