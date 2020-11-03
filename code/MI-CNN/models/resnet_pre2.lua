require 'nn'
require 'torch'
require 'cutorch'
require 'cunn'
require 'cudnn'

local function createModel(opt)
   -- local res_pretrained = 'models/pretrained/model_best_res50_3036c.t7' -- exclude softmax
   -- local res_pretrained = 'models/pretrained/model_best_res50_51c_vmmr.t7' -- exclude softmax
   -- local res_pretrained = 'models/pretrained/model_best_res50_51c_merged.t7' -- exclude softmax
   local res_pretrained = 'models/pretrained/model_best_res50_51c_comp.t7'
   -- local res_pretrained = 'models/pretrained/resnet-50.t7'
   -- local res_pretrained = 'models/pretrained/model_best_res50_3036c.t7'
   local inputSize = 224
   local n_features = 2048
   local n_classes = 51

   local resnet = torch.load(res_pretrained)
   -- resnet:remove(11)
   -- resnet:add(nn.Linear(n_features, n_classes))
   -- --resnet:insert(nn.View(1,3,inputSize, inputSize):setNumInputDims(3),1)
   -- resnet:add(nn.ReLU(true))
   -- resnet:add(nn.MulConstant( -0.001))
   -- resnet:add(nn.Exp())
   --local mapping = nn.MapTable():add(resnet)
   local netsum = nn.Sequential()
            --:add(nn.Square())
            --:add(nn.TemporalConvolution(51, 51, 1, 1))
            --:add(nn.Tanh())
            :add(nn.Sum(1,2, 1))
   local netmax = nn.Sequential()
                  :add(nn.ReLU(true))
                  --:add(nn.TemporalConvolution(51, 51, 1, 1))
                  --:add(nn.Sigmoid())
                  :add(nn.Max(1,2))

   local weighting = nn.Sequential()
         :add(nn.ConcatTable()
            :add(netsum)
            --:add(nn.Max(1,2))
            :add(netmax)
            )
         :add(nn.CMulTable(true))

   local model = nn.Sequential()
   model:add(resnet)
   model:add(weighting)
   -- model:add(nn.SplitTable(1, 2))
   -- model:add(nn.CMulTable())
   -- model:add(nn.MulConstant(-1.0))
   -- model:add(nn.AddConstant(1.0))
   model:add(nn.View(1,-1))
   -- model:add(nn.Linear(n_classes, n_classes))
   model:add(nn.SoftMax())
   -- model:add(nn.Log())
   -- model:add(nn.LogSoftMax())
   model:cuda()

   return model
end

return createModel

-- test
--[[
local model = createModel(1):cuda()
print(model)
local input = torch.FloatTensor(6,3,224,224):fill(1):cuda()
local output = model:forward(input)
print(output:size())
--]]
