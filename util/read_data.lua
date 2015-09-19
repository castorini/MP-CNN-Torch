--[[

  Functions for loading data from disk.

--]]

function similarityMeasure.read_embedding(vocab_path, emb_path)
  local vocab = similarityMeasure.Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

function similarityMeasure.read_sentences(path, vocab)
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
    end
    sentences[#sentences + 1] = sent
  end

  file:close()
  return sentences
end

function similarityMeasure.read_relatedness_dataset(dir, vocab, task)
  local dataset = {}
  dataset.vocab = vocab
  dataset.lsents = similarityMeasure.read_sentences(dir .. 'a.toks', vocab)
  dataset.rsents = similarityMeasure.read_sentences(dir .. 'b.toks', vocab)
  dataset.size = #dataset.lsents
  local id_file = torch.DiskFile(dir .. 'id.txt')
  local sim_file = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids = torch.IntTensor(dataset.size)
  dataset.labels = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    dataset.ids[i] = id_file:readInt()
    if task == 'sic' then
    	dataset.labels[i] = 0.25 * (sim_file:readDouble() - 1) -- sic data
    elseif task == 'vid' then
	dataset.labels[i] = 0.2 * (sim_file:readDouble()) -- vid data
    else
    	dataset.labels[i] = (sim_file:readDouble()) -- twi and msp
    end
  end
  id_file:close()
  sim_file:close()
  return dataset
end

