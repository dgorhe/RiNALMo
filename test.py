import torch
from rinalmo.pretrained import get_pretrained_model
from rinalmo.inference_device import select_inference_device, inference_autocast


if __name__ == "__main__":
    DEVICE = select_inference_device()

    model, alphabet = get_pretrained_model(model_name="giga-v1", device=DEVICE)
    model.eval()
    seqs = ["ACUUUGGCCA", "CCCGGU"]

    tokens = torch.tensor(alphabet.batch_tokenize(
        seqs), dtype=torch.int64, device=DEVICE)
    with torch.no_grad(), inference_autocast(DEVICE):
        outputs = model(tokens)

    print(outputs["representation"])
