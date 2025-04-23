# ARM GNU Toolchain

## Comamnds

### Assembler
```bash
third_party/arm/arm-gnu-toolchain-11.3.rel1-darwin-x86_64-arm-none-eabi/bin/arm-none-eabi-as -mcpu=cortex-m0 -march=armv6s-m -mthumb scum/firmware/common/startup.s -o scum/firmware/common/startup.o

arm-none-eabi-as -march=armv6-m -mthumb -gdwarf-2 -mthumb-interwork --MD startup.d -I/Applications/ARM/arm-none-eabi/include -o startup.o ../../common/startup.s
```

### Compiler
```bash
third_party/arm/arm-gnu-toolchain-11.3.rel1-darwin-x86_64-arm-none-eabi/bin/arm-none-eabi-gcc -Wall -Werror -O0 -nostdlib -nostartfiles -ffreestanding -mthumb -mcpu=cortex-m0 -march=armv6s-m -I scum/firmware/common/ -c scum/firmware/applications/hello_world/hello_world.c -o scum/firmware/applications/hello_world/hello_world.o

arm-none-eabi-gcc -c -march=armv6-m -mthumb -gdwarf-2 -MD -Wall -Werror -O0 -mapcs-frame -mthumb-interwork -I/Applications/ARM/arm-none-eabi/include -D__UVISION_VERSION="525" -DARMCM0 -o hello_world.o hello_world.c
arm-none-eabi-gcc -c -march=armv6-m -mthumb -gdwarf-2 -MD -Wall -Werror -O0 -mapcs-frame -mthumb-interwork -I/Applications/ARM/arm-none-eabi/include -D__UVISION_VERSION="525" -DARMCM0 -o retarget.o ../../common/retarget.c
```

### Linker
```bash
third_party/arm/arm-gnu-toolchain-11.3.rel1-darwin-x86_64-arm-none-eabi/bin/arm-none-eabi-ld -A=armv6-m -e Reset_Handler -o scum/firmware/applications/hello_world/hello_world -Map scum/firmware/applications/hello_world/hello_world.map scum/firmware/applications/hello_world/hello_world.o scum/firmware/common/*.o third_party/arm/arm-gnu-toolchain-11.3.rel1-darwin-x86_64-arm-none-eabi/arm-none-eabi/lib/thumb/v6-m/nofp/*.a third_party/arm/arm-gnu-toolchain-11.3.rel1-darwin-x86_64-arm-none-eabi/lib/gcc/arm-none-eabi/11.3.1/thumb/v6-m/nofp/*.a

arm-none-eabi-ld -mthumb-interwork -e Reset_Handler -Map hello_world.map -o hello_world *.o -nostartfiles -lm
arm-none-eabi-ld -A=armv6-m -e Reset_Handler -Map hello_world.map -o hello_world hello_world.o *.o /Applications/ARM/arm-none-eabi/lib/thumb/v6-m/nofp/*.a /Applications/ARM/lib/gcc/arm-none-eabi/10.3.1/thumb/v6-m/nofp/*.a -lm
arm-none-eabi-ld -A=armv6-m -e Reset_Handler -Map hello_world.map -o hello_world *.o /ARM/ARMCC/lib/armlib/*.l -lm
```

### Disassembler
```bash
third_party/arm/arm-gnu-toolchain-11.3.rel1-darwin-x86_64-arm-none-eabi/bin/arm-none-eabi-objdump -D scum/firmware/applications/hello_world/hello_world.o
```

### Binary
```bash
third_party/arm/arm-gnu-toolchain-11.3.rel1-darwin-x86_64-arm-none-eabi/bin/arm-none-eabi-objcopy -O binary scum/firmware/applications/hello_world/hello_world scum/firmware/applications/hello_world/hello_world.bin
```
