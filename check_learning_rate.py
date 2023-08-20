import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir  = os.path.dirname(currentdir)
sys.path.append(parentdir)
from common import *

CFG = {
    'image_size': 10,
    'learning_rate' : 65e-5,
    'scheduler_name': 'OneCycleLR',

    'T_0': 61, 
    'T_max': 10,
    'T_mult': 2,
    'min_lr': 1e-6,
    'max_lr': 0.001,
    'pct_start': 0.3,
    'lr_div' : 25,
    'final_div_factor': 1000.0,


    'no_batches': 782,
    'batch_size': 4,

    'warmup_epochs': 1,
    'cosine_epochs': 6,
    'epochs': 100,

    'update_per_batch': True,
    'print_freq': 500
}

def get_scheduler(optimizer, scheduler_params = CFG):
    if scheduler_params['scheduler_name'] == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0              = scheduler_params['T_0'],
            T_mult           = scheduler_params['T_mult'],
            eta_min          = scheduler_params['min_lr'],
            last_epoch       = -1,
        )
    elif scheduler_params['scheduler_name'] == 'OneCycleLR':
        scheduler = OneCycleLR(
            optimizer,
            max_lr           = scheduler_params['max_lr'],
            steps_per_epoch  = scheduler_params['no_batches'],
            epochs           = scheduler_params['epochs'],
            pct_start        = scheduler_params['pct_start'],
            div_factor       = scheduler_params['lr_div'],
            final_div_factor = scheduler_params['final_div_factor']
        )
    elif scheduler_params['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max          = scheduler_params['T_max'],
            eta_min        = scheduler_params['min_lr'],
            last_epoch     = -1
        )
    elif scheduler_params["scheduler_name"] == "GradualWarmupSchedulerV2 + CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, scheduler_params["cosine_epochs"])
        scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier = scheduler_params['T_mult'], total_epoch = scheduler_params["warmup_epochs"], after_scheduler = scheduler)
        return scheduler_warmup

    return scheduler


dataset = datasets.FakeData(size = CFG['no_batches'] * CFG['batch_size'], transform = transforms.Compose([transforms.Resize(CFG['image_size']), transforms.ToTensor()]))

loader = DataLoader(
    dataset,
    batch_size   = CFG['batch_size'],
    shuffle      = False,
    num_workers  = 0, 
    drop_last    = True
)

RANK = 0
DEVICE = torch.device('cuda:{}'.format(RANK) if torch.cuda.is_available() else 'cpu')

model     = nn.Linear(3 * CFG['image_size'] * CFG['image_size'], 10).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr = CFG['learning_rate'])
scheduler = get_scheduler(optimizer, CFG)
criterion = nn.NLLLoss()

lrs = []
for epoch in range(CFG['epochs']):
    print(f"EPOCH: {epoch}")
    for step, (data, target) in enumerate(loader):
        if step % CFG['print_freq'] < 10 or step > (len(loader) - 10):
            print('[Epoch]: {}, [Batch]: {}, [LR]: {}'.format(
                epoch, step, np.round(scheduler.get_last_lr()[0], 8)))
            
        lrs.append(optimizer.param_groups[0]['lr'])
        #lrs.append(scheduler.get_last_lr())
        data   = data.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data.view(CFG['batch_size'], -1))
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()

        if CFG['update_per_batch'] == True: scheduler.step()

    if CFG['update_per_batch'] == False: scheduler.step()

xcoords = [CFG['no_batches'] * x for x in range(CFG['epochs'])]

plt.figure(figsize = (18, 10))
for xc in xcoords:
    plt.axvline(x = xc, color = 'red')

plt.plot(lrs)
plt.title("Learning Rate Scheduling", color = 'r')
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.xticks(xcoords, list(range(CFG['epochs'])))
plt.savefig("learning_rate_scheduling.png")
plt.show()