// based on https://ffmpeg.org/doxygen/trunk/remuxing_8c-example.html

#include "TransformVideo.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/dnn_superres.hpp>

#include <functional>
#include <iostream>

const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{@input         |      | input file           }"
    "{@output        |      | output file          }"
    "{m model      | fsrcnn | desired model name   }"
    "{f file | FSRCNN_x2.pb | desired model file   }"
    "{u upscale      | 2    | upscale factor       }"
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

        const auto modelName = parser.get<cv::String>("m");
        const auto modelFile = parser.get<cv::String>("f");
        const auto upscale = parser.get<int>("u");

        //Create the module's object
        cv::dnn_superres::DnnSuperResImpl sr;

        //Read the desired model
        sr.readModel(modelFile);

        //Set the desired model and scale to get correct pre- and post-processing
        sr.setModel(modelName, upscale);


        auto lam = [&sr](cv::Mat& img) {
            //Upscale
            cv::Mat img_new;
            sr.upsample(img, img_new);
            img = img_new;
        };

        return TransformVideo(input.c_str(), output.c_str(), lam, upscale);
    }
    catch (const std::exception& ex) {
        std::cerr << "Exception " << typeid(ex).name() << ": " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
