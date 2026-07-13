
import json, time, sys

path = '/home/ubuntu/harsha/dreamzero/checkpoints/run_10_july/loss_log.jsonl'
patience = 1000
threshold = 0.001
best_loss = float('inf')
steps_without_improvement = 0

while True:
    try:
        entries = [json.loads(l) for l in open(path) if l.strip()]
        if entries:
            last = entries[-1]
            loss = last.get('loss', float('inf'))
            step = last.get('step', 0)
            if best_loss - loss > threshold:
                best_loss = loss
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
            print(f'Step {step}: loss={loss:.4f}, best={best_loss:.4f}, no_improve={steps_without_improvement}/{patience}')
            if steps_without_improvement >= patience:
                print(f'Loss plateaued for {patience} steps — consider stopping training')
    except FileNotFoundError:
        print('Waiting for loss_log.jsonl...')
    time.sleep(30)
