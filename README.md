### Towards Unified and High-Fidelity Multi-Modality Image Fusion Model for Adverse Weather via Vision-Language Model
##  Network Architecture
![](./figs/overview.png)

## Contents
- [Quick Run](#Quick-Run)
- [Adverse Weather Fusion](#Adverse-Weather-Fusion)
- [Gallery](#Gallery)



<h2 id="Quick-Run"> Quick Run</h2>

Run( An example of Haze.)
```
python test.py \
 --ir_path ./Test_imgs/Haze/ir \
 --vi_path ./Test_imgs/Haze/vi \
 --weights_path ./checkpoint/AWM_Fuse.pth \
 --save_path ./result/Haze  \
 --input_text ./Test_imgs/Haze/Haze_captions \
 --blip_vi_text ./Test_imgs/Haze/vi_npy \
 --blip_ir_text ./Test_imgs/Haze/ir_npy
```

<h2 id='Adverse-Weather-Fusion'> Adverse Weather Fusion</h2>

<table border="" cellspacing="0" cellpadding="0">
  <tr>
    <td align="center"><b>Rain</b></td>
    <td align="center"><b>Haze</b></td>
    <td align="center"><b>Snow</b></td>
  </tr>
    <tr>
    <td style="padding: 20px;"><img src="./figs/rain_removal_gradient.gif" alt="Rain" width="240" height="180"></td>
    <td style="padding: 20px;"><img src="./figs/haze_removal_gradient.gif" alt="Haze" width="240" height="180"></td>
    <td style="padding: 20px;"><img src="./figs/snow_removal_gradient.gif" alt="Snow" width="240" height="180"></td>
  </tr>
</table>


<h2 id='Gallery'> Gallery</h2>

![Gallery](./figs/Rain.png)
Fig1. Comparison of fusion results obtained by the proposed algorithm under rain weather and the results of the comparison methods under ideal condition.

![Gallery](./figs/Haze.png)
Fig2. Comparison of fusion results obtained by the proposed algorithm under haze weather and the results of the comparison methods under ideal condition.

![Gallery](./figs/Snow.png)
Fig3. Comparison of fusion results obtained by the proposed algorithm under snow weather, and the results of the comparison methods under ideal condition.




