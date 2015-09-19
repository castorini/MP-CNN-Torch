--[[
  Training script for semantic relatedness prediction on the SICK dataset.
  We Thank Kaitai Change to provide a basis code for this. Our modifications are based on his.
--]]

require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

similarityMeasure = {}

include('util/read_data.lua')
include('util/Vocab.lua')
include('Conv.lua')
include('CsDis.lua')
--include('PaddingReshape.lua')
printf = utils.printf

-- global paths (modify if desired)
similarityMeasure.data_dir        = 'data'
similarityMeasure.models_dir      = 'trained_models'
similarityMeasure.predictions_dir = 'predictions'

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end

-- Pearson correlation
function pearson(x, y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm())
end

-- read command line arguments
local args = lapp [[
Training script for semantic relatedness prediction on the SICK dataset.
  -m,--model  (default dependency) Model architecture: [dependency, lstm, bilstm]
  -l,--layers (default 1)          Number of layers (ignored for Tree-LSTM)
  -d,--dim    (default 150)        LSTM memory dimension
]]

local model_name, model_class, model_structure
model_name = 'convOnly'
model_class = similarityMeasure.Conv
model_structure = model_name

torch.seed()
--torch.manualSeed(123)
print('<torch> using the specified seed: ' .. torch.initialSeed())

-- directory containing dataset files
local data_dir = 'data/msrvid/'

-- load vocab
local vocab = similarityMeasure.Vocab(data_dir .. 'vocab-cased.txt')

-- load embeddings
print('loading word embeddings')

local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = similarityMeasure.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')

local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()
local taskD = 'vid'
-- load datasets
print('loading datasets')
local train_dir = data_dir .. 'train/'
local dev_dir = data_dir .. 'dev/'
local test_dir = data_dir .. 'test/'
local train_dataset = similarityMeasure.read_relatedness_dataset(train_dir, vocab, taskD)
local dev_dataset = similarityMeasure.read_relatedness_dataset(dev_dir, vocab, taskD)
printf('num train = %d\n', train_dataset.size)
printf('num dev   = %d\n', dev_dataset.size)

-- initialize model
local model = model_class{
  emb_vecs   = vecs,
  structure  = model_structure,
  mem_dim    = 150,
  task       = taskD,
}

-- number of epochs to train
local num_epochs = 35

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()


if lfs.attributes(similarityMeasure.predictions_dir) == nil then
  lfs.mkdir(similarityMeasure.predictions_dir)
end

-- train
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model

-- threads
--torch.setnumthreads(4)
--print('<torch> number of threads in used: ' .. torch.getnumthreads())

header('Training model')

local id = 2007
print("Id: " .. id)
for i = 1, num_epochs do
  local start = sys.clock()
  print('--------------- EPOCH ' .. i .. '--- -------------')
  model:trainCombineOnly(train_dataset)
  print('Finished epoch in ' .. ( sys.clock() - start) )

  local dev_predictions = model:predict_dataset(dev_dataset)
  local dev_score = pearson(dev_predictions, dev_dataset.labels)
  printf('-- score: %.5f\n', dev_score)
end
print('finished training in ' .. (sys.clock() - train_start))

