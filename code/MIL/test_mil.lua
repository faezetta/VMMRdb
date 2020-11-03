--- packages
require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'
require 'paths'
require 'xlua'
t=require 'datasets/transforms'
csv = require 'csvigo'
torch.setdefaulttensortype('torch.FloatTensor')

-- Load model
model = torch.load('checkpoints/milresnet/model_best.t7'):cuda()
-- Evaluate mode
model:evaluate()


-- preprocess
meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
}

-- load data
function load_testdata()
    local base_dir ='/home/keishin/sandbox/CARS/VMMRdb_Resnet/val/'
    local data = {}
    local idx=1
    for dir in paths.iterdirs(base_dir) do
        for f in paths.files(base_dir..dir..'/','.jpg')do 
            data[idx] ={base_dir..dir..'/'..f, dir, f}
            idx = idx + 1
        end
    end
    return data
end

--[[
function load_testdata_flat()
    local base_dir ='FloydWarnock2/'
    local data = {}
    for f in paths.files(base_dir,'.jpg')do 
        table.insert(data, {paths.concat(base_dir, f), f})
    end
    return data
end
--]]

validation = load_testdata()
print("# test samples",#validation)
print(validation[1])

-- label names
temp = torch.load('gen/imagenet.t7')
indexToClass = temp.classList

--check
--print(#indexToClass)
--print(indexToClass[10])


local N = 3036
csvf = csv.File('Res50_mil.csv', "w", ',')

time = sys.clock()
-- csv header
header ={}
table.insert(header, 'Name')
table.insert(header, 'Grand Truth')
table.insert(header, 'Pred Label')
table.insert(header, 'Prob_Pred_Label')
for i=1,N do
    table.insert(header, indexToClass[i])
end
--print(header)
csvf:write(header)

-- predict and write to csv

-- check
--for i=1, 10 do
for i=1,#validation do
    
   -- load the image as a RGB float tensor with values 0..1
   local img = image.load(validation[i][1], 3, 'float')
    
   -- Scale, normalize, and crop the image
   img = transform(img)
   
   -- generate instances
   local n_insta = 24
   local bag = torch.FloatTensor(n_insta, 3, 224, 224)
   local converter = t.RandomSizedCrop(168)
   local scaler = t.Scale(224)
   local cropper = t.RandomCrop(224,0)
   -- image.save('og.jpg',batch[1]:resize(3, batch[1]:size(3), batch[1]:size(4)))
   -- bag[1] = scaler(temp)
   --image.save('instance-' .. tostring(1) .. '.jpg',bag[1])
   for i = 1, n_insta do
      local temp = converter(img)
      bag[i] = scaler(temp)
      -- print(bag[i]:size())
      -- image.save('instance-' .. tostring(i) .. '.jpg',bag[i])
   end
   local input = bag:view(n_insta, 3, 224, 224)

   
    
   -- View as mini-batch of size 1
   --local batch = img:view(1, table.unpack(img:size():totable()))
    
   -- Get the output
   local output = model:forward(input:cuda()):squeeze()
 
   -- prep probabilities
   local y, pred = torch.max(output,1)
    line = {}
    table.insert(line, validation[i][3])
    table.insert(line, validation[i][2]) 
    --table.insert(line, '')  
    --get acctual label of compcar
    table.insert(line, indexToClass[pred[1]])
    table.insert(line, tostring(y:float()[1]))
    for i=1, output:size(1) do
        table.insert(line, tostring(output[i]))
    end
    
    -- write
    csvf:write(line)
    
    --- intermidiate result
    xlua.progress(i, #validation)

end

csvf:close()
print(sys.clock() - time)
