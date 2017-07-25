--[[
  Author: Hua He
  Usage： th testRun.lua
  Training script for semantic relatedness prediction on the SICK dataset. 
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

--torch.seed()
torch.manualSeed(-3.0753778015266e+18)
--print('<torch> using the automatic seed: ' .. torch.initialSeed())

-- directory containing dataset files
local data_dir = 'data/sick/'

-- load vocab
local vocab = similarityMeasure.Vocab(data_dir .. 'vocab-cased.txt')

-- load embeddings
print('loading word embeddings')

local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = similarityMeasure.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')

local emb_dim = emb_vecs:size(2)

collectgarbage()

local taskD = 'sic'
-- load datasets
local test_dir = data_dir .. 'test/'
local test_dataset = similarityMeasure.read_relatedness_dataset(test_dir, vocab, taskD)
printf('num test  = %d\n', test_dataset.size)

-- initialize model
modelTrained = torch.load("modelSTS.trained.th", 'ascii')
modelTrained.convModel:evaluate()
modelTrained.softMaxC:evaluate()

-- Predict dataset with existing models
for i = 1, test_dataset.size do
    local lsent, rsent = test_dataset.lsents[i], test_dataset.rsents[i]
    local linputs = emb_vecs:index(1, lsent:long()):double()
    local rinputs = emb_vecs:index(1, rsent:long()):double()
    local part2 = modelTrained.convModel:forward({linputs, rinputs})
    local output = modelTrained.softMaxC:forward(part2)
    local val = torch.range(0, 5, 1):dot(output:exp()) 
    print(val)
end



