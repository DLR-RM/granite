/**
MIT License

This file is part of the Granite project which is based on Basalt.
https://github.com/DLR-RM/granite

Copyright (c) Martin Wudenka, Deutsches Zentrum für Luft- und Raumfahrt

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/**
Original license of Basalt:
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <granite/io/marg_data_io.h>

#include <granite/serialization/headers_serialization.h>
#include <granite/utils/filesystem.h>

namespace granite {

MargDataSaver::MargDataSaver(const std::string& path) {
  fs::remove_all(path);
  fs::create_directory(path);

  save_image_queue.set_capacity(300);

  std::string img_path = path + "/images/";
  fs::create_directory(img_path);

  in_marg_queue.set_capacity(1000);

  auto save_func = [&, path]() {
    granite::MargData::Ptr data;

    std::unordered_set<int64_t> processed_opt_flow;

    while (!should_quit) {
      in_marg_queue.pop(data);

      if (data.get()) {
        int64_t kf_id = *data->kfs_to_marg.begin();

        std::string p = path + "/" + std::to_string(kf_id) + ".cereal";
        std::ofstream os(p, std::ios::binary);

        {
          cereal::BinaryOutputArchive archive(os);
          archive(*data);
        }
        os.close();

        for (const auto& d : data->opt_flow_res) {
          if (processed_opt_flow.count(d->t_ns) == 0) {
            save_image_queue.push(d);
            processed_opt_flow.emplace(d->t_ns);
          }
        }

      } else {
        save_image_queue.push(nullptr);
        break;
      }
    }

    std::cout << "Finished MargDataSaver" << std::endl;
  };

  auto save_image_func = [&, img_path]() {
    granite::OpticalFlowResult::Ptr data;

    while (!should_quit) {
      save_image_queue.pop(data);

      if (data.get()) {
        std::string p = img_path + "/" + std::to_string(data->t_ns) + ".cereal";
        std::ofstream os(p, std::ios::binary);

        {
          cereal::BinaryOutputArchive archive(os);
          archive(data);
        }
        os.close();
      } else {
        break;
      }
    }

    std::cout << "Finished image MargDataSaver" << std::endl;
  };

  saving_thread.reset(new std::thread(save_func));
  saving_img_thread.reset(new std::thread(save_image_func));
}  // namespace granite

MargDataLoader::MargDataLoader() : out_marg_queue(nullptr) {}

void MargDataLoader::start(const std::string& path) {
  if (!fs::exists(path))
    std::cerr << "No marg. data found in " << path << std::endl;

  auto func = [&, path]() {
    std::string img_path = path + "/images/";

    std::unordered_set<uint64_t> saved_images;

    std::map<int64_t, OpticalFlowResult::Ptr> opt_flow_res;

    for (const auto& entry : fs::directory_iterator(img_path)) {
      OpticalFlowResult::Ptr data;
      // std::cout << entry.path() << std::endl;
      std::ifstream is(entry.path(), std::ios::binary);
      {
        cereal::BinaryInputArchive archive(is);
        archive(data);
      }
      is.close();
      opt_flow_res[data->t_ns] = data;
    }

    std::map<int64_t, std::string> filenames;

    for (auto& p : fs::directory_iterator(path)) {
      std::string filename = p.path().filename();
      if (!std::isdigit(filename[0])) continue;

      size_t lastindex = filename.find_last_of(".");
      std::string rawname = filename.substr(0, lastindex);

      int64_t t_ns = std::stol(rawname);

      filenames.emplace(t_ns, filename);
    }

    for (const auto& kv : filenames) {
      granite::MargData::Ptr data(new granite::MargData);

      std::string p = path + "/" + kv.second;
      std::ifstream is(p, std::ios::binary);

      {
        cereal::BinaryInputArchive archive(is);
        archive(*data);
      }
      is.close();

      for (const auto& d : data->kfs_all) {
        data->opt_flow_res.emplace_back(opt_flow_res.at(d));
      }

      out_marg_queue->push(data);
    }

    out_marg_queue->push(nullptr);

    std::cout << "Finished MargDataLoader" << std::endl;
  };

  processing_thread.reset(new std::thread(func));
}
}  // namespace granite

namespace cereal {

template <class Archive>
void serialize(Archive& ar, granite::OpticalFlowResult& m) {
  ar(m.t_ns);
  ar(m.observations);
  ar(m.input_images);
}

template <class Archive>
void serialize(Archive& ar, granite::OpticalFlowInput& m) {
  ar(m.t_ns);
  ar(m.img_data);
}

template <class Archive>
void serialize(Archive& ar, granite::ImageData& m) {
  ar(m.exposure);
  ar(m.img);
}

template <class Archive>
static void serialize(Archive& ar, Eigen::AffineCompact2f& m) {
  ar(m.matrix());
}
}  // namespace cereal
