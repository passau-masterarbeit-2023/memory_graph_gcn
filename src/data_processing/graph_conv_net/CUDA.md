# CUDA


## Installing

Follow the instructions [here | Fedora](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#fedora).

First, you need to identify if everything is compatible. Check for GPUs with `lspci | grep VGA`

```shell
 ❮onyr ★ kenzael❯ ❮~❯❯ lspci | grep VGA
00:02.0 VGA compatible controller: Intel Corporation Alder Lake-P Integrated Graphics Controller (rev 0c)
01:00.0 VGA compatible controller: NVIDIA Corporation GA104 [Geforce RTX 3070 Ti Laptop GPU] (rev a1)
```

0. If kernel headers are available for you current kernel, install them with recommended command ` sudo dnf install kernel-devel-$(uname -r) kernel-headers-$(uname -r)`. If there is nothing, then install the latest available kernel headers with  `sudo dnf install gcc kernel-devel kernel-headers`. This will require you to update you kernel to match the latest version.

Notes for myself: Since I have a multi-partition system, I needed first to install the latest Fedora kernel with `sudo dnf update kernel`. Then, I needed to update EFI for Fedora with `sudo grub2-mkconfig -o /boot/efi/EFI/fedora/grub.cfg` or GRUB with `sudo grub2-mkconfig -o /boot/grub2/grub.cfg`. (I choose EFI, but I leave the other command there just in case...). Then I had to reboot to TUXEDO (Ubuntu) OS, since the main GRUB is there (first installed OS). You can edit the GRUB config file with `sudo nano /etc/default/grub`, and then update GRUB `sudo update-grub`. It should be written that Fedora is detected as a foreign OS during the process.

1. First, check that kernel matched the kernel-headers correctly, after previous step.

```shell
 ❮onyr ★ kenzael❯ ❮~❯❯ uname -r
6.4.8-100.fc37.x86_64
 ❮onyr ★ kenzael❯ ❮~❯❯ rpm -q kernel-devel
kernel-devel-6.4.8-100.fc37.x86_64
```
