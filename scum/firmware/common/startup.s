    .syntax     unified
    .arch       armv6-m

    .section    .stack, "w"
    .balign     16
    .equ        Stack_Size, 0x00000800      // 4 KB for the stack.

    .global     __stack_limit
    .global     __stack_top
__stack_limit:
Stack_Mem:
    .org        Stack_Mem + Stack_Size
    .size       __stack_limit, . - __stack_limit
__stack_top:
    .size       __stack_top, . - __stack_top

    .section    .heap, "w"
    .balign     16
    .equ        Heap_Size, 0x00000400       // 2 KB for the heap.

    .global     __heap_base
    .global     __heap_limit
__heap_base:
Heap_Mem:
    .org       Heap_Mem + Heap_Size
    .size       __heap_base, . - __heap_base
__heap_limit:
    .size       __heap_limit, . - __heap_limit

    .eabi_attribute Tag_ABI_align_preserved, 1
    .thumb
    .section    .vectors
    .global __Vectors
__Vectors:
    .word       __stack_top
    .word       Reset_Handler
    .word       0
    .word       0
    .word       0
    .word       0
    .word       0
    .word       0
    .word       0
    .word       0
    .word       0
    .word       0
    .word       0
    .word       0
    .word       0
    .word       0
    // External interrupts.
    .word       UART_Handler
    .word       EXT_GPIO3_ACTIVEHIGH_DEBOUNCED_Handler
    .word       EXT_OPTICAL_IRQ_IN_Handler
    .word       ADC_Handler
    .word       0
    .word       0
    .word       RF_Handler
    .word       RFTIMER_Handler
    .word       RAWCHIPS_STARTVAL_Handler
    .word       RAWCHIPS_32_Handler
    .word       0
    .word       OPTICAL_SFD_Handler
    .word       EXT_GPIO8_ACTIVEHIGH_Handler
    .word       EXT_GPIO9_ACTIVELOW_Handler
    .word       EXT_GPIO10_ACTIVELOW_Handler
    .word       0
    .size       __Vectors, . - __Vectors

    .section    .text, "ax"
    .balign     8

    .global     Reset_Handler
    .type       Reset_Handler, "function"
Reset_Handler:
    ldr         r1, =0xE000E100             // Interrupt set enable register.
    ldr         r0, =0x00000000             // Remember to enable the interrupts.
    str         r0, [r1]

    .global     main
    ldr         r0, =main
    bx          r0                          // Branch to main.
    .size       Reset_Handler, . - Reset_Handler

    .global     UART_Handler
    .type       UART_Handler, "function"
UART_Handler:
    .global     uart_rx_isr

    push        {r0, lr}
    movs        r0, #1                      // Mask all interrupts.
    msr         primask, r0

    bl          uart_rx_isr

    movs        r0, #0                      // Enable all interrupts.
    msr         primask, r0
    pop         {r0, pc}
    .size       UART_Handler, . - UART_Handler

    .global     ADC_Handler
    .type       ADC_Handler, "function"
ADC_Handler:
    .global     adc_isr


    push        {r0, lr}
    movs        r0, #1                      // Mask all interrupts.
    msr         primask, r0

    bl          adc_isr

    movs        r0, #0                      // Enable all interrupts.
    msr         primask, r0
    pop         {r0, pc}
    .size       ADC_Handler, . - ADC_Handler

    .global     RF_Handler
    .type       RF_Handler, "function"
RF_Handler:
    .global     radio_isr

    push        {r0, lr}
    movs        r0, #1                      // Mask all interrupts.
    msr         primask, r0

    bl          radio_isr

    movs        r0, #0                      // Enable all interrupts.
    msr         primask, r0
    pop         {r0, pc}
    .size       RF_Handler, . - RF_Handler

    .global     RFTIMER_Handler
    .type       RFTIMER_Handler, "function"
RFTIMER_Handler:
    .global     rftimer_isr

    push        {r0, lr}
    movs        r0, #1                      // Mask all interrupts.
    msr         primask, r0

    bl          rftimer_isr

    movs        r0, #0                      // Enable all interrupts.
    msr         primask, r0
    pop         {r0, pc}
    .size       RFTIMER_Handler, . - RFTIMER_Handler

    .global     EXT_GPIO3_ACTIVEHIGH_DEBOUNCED_Handler
    .type       EXT_GPIO3_ACTIVEHIGH_DEBOUNCED_Handler, "function"
EXT_GPIO3_ACTIVEHIGH_DEBOUNCED_Handler:
    .global     ext_gpio3_activehigh_debounced_isr

    push        {r0, lr}
    movs        r0, #1                      // Mask all interrupts.
    msr         primask, r0

    bl          ext_gpio3_activehigh_debounced_isr

    movs        r0, #0                      // Enable all interrupts.
    msr         primask, r0
    pop         {r0, pc}
    .size       EXT_GPIO3_ACTIVEHIGH_DEBOUNCED_Handler, . - EXT_GPIO3_ACTIVEHIGH_DEBOUNCED_Handler

    .global     EXT_GPIO8_ACTIVEHIGH_Handler
    .type       EXT_GPIO8_ACTIVEHIGH_Handler, "function"
EXT_GPIO8_ACTIVEHIGH_Handler:
    .global     ext_gpio8_activehigh_isr

    push        {r0, lr}
    movs        r0, #1                      // Mask all interrupts.
    msr         primask, r0

    bl          ext_gpio8_activehigh_isr

    movs        r0, #0                      // Enable all interrupts.
    msr         primask, r0
    pop         {r0, pc}
    .size       EXT_GPIO8_ACTIVEHIGH_Handler, . - EXT_GPIO8_ACTIVEHIGH_Handler

    .global     EXT_GPIO9_ACTIVELOW_Handler
    .type       EXT_GPIO9_ACTIVELOW_Handler, "function"
EXT_GPIO9_ACTIVELOW_Handler:
    .global     ext_gpio9_activelow_isr

    push        {r0, lr}
    movs        r0, #1                      // Mask all interrupts.
    msr         primask, r0

    bl          ext_gpio9_activelow_isr

    movs        r0, #0                      // Enable all interrupts.
    msr         primask, r0
    pop         {r0, pc}
    .size       EXT_GPIO9_ACTIVELOW_Handler, . - EXT_GPIO9_ACTIVELOW_Handler

    .global     EXT_GPIO10_ACTIVELOW_Handler
    .type       EXT_GPIO10_ACTIVELOW_Handler, "function"
EXT_GPIO10_ACTIVELOW_Handler:
    .global     ext_gpio10_activelow_isr

    push        {r0, lr}
    movs        r0, #1                      // Mask all interrupts.
    msr         primask, r0

    bl          ext_gpio10_activelow_isr

    movs        r0, #0                      // Enable all interrupts.
    msr         primask, r0
    pop         {r0, pc}
    .size       EXT_GPIO10_ACTIVELOW_Handler, . - EXT_GPIO10_ACTIVELOW_Handler

    .global     RAWCHIPS_STARTVAL_Handler
    .type       RAWCHIPS_STARTVAL_Handler, "function"
RAWCHIPS_STARTVAL_Handler:
    .global     rawchips_startval_isr

    push        {r0, lr}
    movs        r0, #1                      // Mask all interrupts.
    msr         primask, r0

    bl          rawchips_startval_isr

    movs        r0, #0                      // Enable all interrupts.
    msr         primask, r0
    pop         {r0, pc}
    .size       RAWCHIPS_STARTVAL_Handler, . - RAWCHIPS_STARTVAL_Handler

    .global     RAWCHIPS_32_Handler
    .type       RAWCHIPS_32_Handler, "function"
RAWCHIPS_32_Handler:
    .global     rawchips_32_isr

    push        {r0, lr}
    movs        r0, #1                      // Mask all interrupts.
    msr         primask, r0

    bl          rawchips_32_isr

    movs        r0, #0                      // Enable all interrupts.
    msr         primask, r0
    pop         {r0, pc}
    .size       RAWCHIPS_32_Handler, . - RAWCHIPS_32_Handler

    .global     EXT_OPTICAL_IRQ_IN_Handler
    .type       EXT_OPTICAL_IRQ_IN_Handler, "function"
EXT_OPTICAL_IRQ_IN_Handler:
    .global     optical_32_isr

    push        {r0, lr}
    movs        r0, #1                      // Mask all interrupts.
    msr         primask, r0

    bl          optical_32_isr

    movs        r0, #0                      // Enable all interrupts.
    msr         primask, r0
    pop         {r0, pc}
    .size       EXT_OPTICAL_IRQ_IN_Handler, . - EXT_OPTICAL_IRQ_IN_Handler

    .global     OPTICAL_SFD_Handler
    .type       OPTICAL_SFD_Handler, "function"
OPTICAL_SFD_Handler:
    .global     optical_sfd_isr

    push        {r0, lr}
    movs        r0, #1                      // Mask all interrupts.
    msr         primask, r0

    bl          optical_sfd_isr

    movs        r0, #0                      // Enable all interrupts.
    msr         primask, r0
    pop         {r0, pc}
    .size       OPTICAL_SFD_Handler, . - OPTICAL_SFD_Handler

    .balign     4

// Initial stack and heap.
    .ifndef     __MICROLIB
    .global     __use_two_region_memory
    .global     __user_initial_stackheap
__user_initial_stackheap:
    ldr         r0, =Heap_Mem
    ldr         r1, =(Stack_Mem + Stack_Size)
    ldr         r2, =(Heap_Mem + Heap_Size)
    ldr         r3, =Stack_Mem
    bx          lr

    .align
    .endif

    .end
