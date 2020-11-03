--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')
local t = require 'datasets/transforms'

require 'image' -- added
-- ffi = require 'ffi' -- added
-- require 'paths' --added
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   for i, split in ipairs{'train', 'val'} do
      local dataset = datasets.create(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split)
   end

   return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
   local manualSeed = opt.manualSeed
   local function init()
      require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      return dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.nCrops = (split == 'val' and opt.tenCrop) and 10 or 1
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchSize = math.floor(opt.batchSize / self.nCrops)
   local function getCPUType(tensorType)
      if tensorType == 'torch.CudaHalfTensor' then
         return 'HalfTensor'
      elseif tensorType == 'torch.CudaDoubleTensor' then
         return 'DoubleTensor'
      else
         return 'FloatTensor'
      end
   end
   self.cpuType = getCPUType(opt.tensorType)
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local perm = torch.randperm(size)
   local split = self.split

   local idx, sample = 1, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices, nCrops, cpuType)
               local sz = indices:size(1)
               local batch, imageSize
               local target = torch.IntTensor(sz)
               local sample_instance
               local bag
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx)
                  local input = _G.preprocess(sample.input)
                  if not batch then
                     imageSize = input:size():totable()
                     --print('imageSize',table.unpack(imageSize))
                     if nCrops > 1 then table.remove(imageSize, 1) end
                     batch = torch[cpuType](sz, nCrops, table.unpack(imageSize))
                  end
                  batch[i]:copy(input)
                  target[i] = sample.target

                  --- 2 ADDED muitiple regios ---
                  local sample_instance_label = _G.dataset:get_instances(idx)
                  sample_instance = sample_instance_label.input
                  local prep
                  if split == 'train' then
                     prep = t.Compose
                                 {
                                    t.ColorJitter({
                                       brightness = 0.4,
                                       contrast = 0.4,
                                       saturation = 0.4,
                                    }),
                                    t.Lighting(0.1, pca.eigval, pca.eigvec),
                                    t.ColorNormalize(meanstd),
                                 }
                  else
                     prep = t.Compose
                                    {
                                       t.Scale(224),
                                       t.ColorNormalize(meanstd),
                                    }
                  end
                  local prob = torch.uniform()
                  local nnn = sample_instance:size(1)
                  for jj=1,nnn do
                     sample_instance[jj] = prep(sample_instance[jj])
                     if split == 'train' and 0.5 < prob then
                        sample_instance[jj] = image.hflip(sample_instance[jj])
                     end
                  end
                  --- t.RandomSizedCrop(256) in imagenet.lua
                  local n_insta = 24
                  bag = torch.FloatTensor(n_insta, 3, 224, 224)
                  local converter = t.RandomSizedCrop(168)
                  local scaler = t.Scale(224)
                  local cropper = t.RandomCrop(224,0)
                  --print('batchsize', batch[1]:size())
                  for k = 1, n_insta do
                     if k <= nnn then
                        bag[k] = sample_instance[k]:clone()
                        --print(i, sample_instance[i]:size())
                     else
                        local temp = converter(batch[1]:resize(3,batch[1]:size(3), batch[1]:size(4)))
                        if split == 'train' and 0.5 < prob then
                           temp = image.hflip(temp)
                        end
                        bag[k] = scaler(temp):clone()
                        -- print(i, scaler(temp):size())
                     end
                     --image.save('instances/temp'..tostring(k)..'.jpg', bag[k])
                     -- print(bag[i]:size())
                     --image.save('instances/instance-' .. tostring(i) .. '.jpg',bag[i])
                  end
                  --- 2 end added -------
               end
               collectgarbage()
               ----- 2 ADDED--------
               return {
                  input = bag:view(24, 3, 224, 224),
                  --input = bag,
                  target = target,
               }
               -- return {
               --    --input = batch:view(-1, 3, 224, 224),
               --    input = sample_instance,
               --    target = target,
               -- }
               ---- 2 ADDED END-----

               --[[
               ------- 1 ADDED -----
               --- t.RandomSizedCrop(256) in imagenet.lua
               local n_insta = 24
               local bag = torch.FloatTensor(n_insta, 3, 224, 224)
               local converter = t.RandomSizedCrop(168)
               local scaler = t.Scale(224)
               local cropper = t.RandomCrop(224,0)
               -- image.save('og.jpg',batch[1]:resize(3, batch[1]:size(3), batch[1]:size(4)))
               local temp = cropper(batch[1]:resize(3,batch[1]:size(3), batch[1]:size(4)))
               -- bag[1] = scaler(temp)
               --image.save('instance-' .. tostring(1) .. '.jpg',bag[1])
               for i = 1, n_insta do
                  local temp = converter(batch[1]:resize(3,batch[1]:size(3), batch[1]:size(4)))
                  bag[i] = scaler(temp)
                  -- print(bag[i]:size())
                  -- image.save('instance-' .. tostring(i) .. '.jpg',bag[i])
               end

               return {
                  input = bag:view(n_insta, 3, 224, 224),
                  target = target,
               }

               -------- 1 END ADDED ------
               --]]

               -- return {
               --    input = batch:view(sz * nCrops, table.unpack(imageSize)),
               --    target = target,
               -- }

            end,
            function(_sample_)
               sample = _sample_
            end,
            indices,
            self.nCrops,
            self.cpuType
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader
