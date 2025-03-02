## 目录

<details>
<summary>概述</summary>

- [概述](#概述)
</details>

<details>
<summary>1.基础知识</summary>

- [1.基础知识](#1基础知识)
  - [一.任务调度器简述](#一任务调度器简述)
    - [1.什么是任务调度器](#1-什么是任务调度器)
    - [2.freertos的调度方式](#2-freertos的调度方式)
    - [3.抢占式调度过程](#3-抢占式调度过程)
    - [4.时间片是什么](#4-时间片是什么)
    - [5.时间片调度过程](#5-时间片调度过程)
  - [二.任务状态](#二任务状态)
    - [1.freertos的任务状态](#1-freertos的任务状态)
    - [2.四种状态之间的转换关系](#2-四种状态之间的转换关系)
    - [3.任务状态列表](#3-任务状态列表)
</details>

<details>
<summary>2.freertos系统配置文件详解</summary>

- [2.freertos系统配置文件详解](#2-freertos系统配置文件详解)
</details>

<details>
<summary>3.任务的创建和删除</summary>

- [3.任务的创建和删除](#3-任务的创建和删除)
  - [一.任务创建和删除API函数](#一任务创建和删除api函数)
    - [1.任务创建和删除的本质](#1-任务创建和删除的本质)
    - [2.任务动态创建和静态创建的区别](#2-任务动态创建和静态创建的区别)
    - [3.任务控制块结构体成员介绍](#3-任务控制块结构体成员介绍)
    - [4.什么是临界保护区](#4-什么是临界保护区)
    - [5.动态创建的优点](#5-动态创建的优点)
    - [6.静态创建的优点](#6-静态创建的优点)
  - [二.任务的创建（动态）](#二任务的创建动态)
    - [1.动态函数的创建](#1-动态函数的创建)
    - [2.什么是句柄](#2-什么是句柄)
    - [3.实现动态创建任务流程](#3-实现动态创建任务流程)
    - [4.动态任务创建函数内部实现简述](#4-动态任务创建函数内部实现简述)
  - [三.任务的创建（静态）](#三任务的创建静态)
    - [1.静态函数的创建](#1-静态函数的创建)
    - [2.实现静态创建任务流程](#2-实现静态创建任务流程)
    - [3.静态任务创建函数内部实现简述](#3-静态任务创建函数内部实现简述)
  - [四.任务的删除](#四任务的删除)
    - [1.任务删除函数](#1-任务删除函数)
    - [2.删除任务流程](#2-删除任务流程)
    - [3.删除任务函数内部实现简述](#3-删除任务函数内部实现简述)
</details>

<details>
<summary>4.任务的挂起和恢复</summary>

- [4.任务的挂起和恢复](#4-任务的挂起和恢复)
  - [一.任务的挂起和恢复介绍](#一任务的挂起和恢复介绍)
  - [二.任务的挂起](#二任务的挂起)
    - [1.挂起函数介绍](#1-挂起函数介绍)
    - [2.任务挂起函数内部实现](#2-任务挂起函数内部实现)
  - [三.任务的恢复](#三任务的恢复)
    - [1.任务恢复函数介绍（任务中）](#1-任务恢复函数介绍任务中)
    - [2.任务回复函数的实现（任务中）](#2-任务回复函数的实现任务中)
    - [3.任务恢复函数介绍（中断中）](#3-任务恢复函数介绍中断中)
    - [4.任务恢复函数内部实现（中断中）](#4-任务恢复函数内部实现中断中)
</details>

<details>
<summary>5.中断管理</summary>

- [5.中断管理](#5-中断管理)
  - [一.中断介绍](#一中断介绍)
    - [1.什么是中断](#1-什么是中断)
    - [2.中断执行机制](#2-中断执行机制)
  - [二.中断优先级分组设置](#二中断优先级分组设置)
    - [1.中断优先级分组介绍](#1-中断优先级分组介绍)
    - [2.什么是去抢占优先级什么是子优先级](#2-什么是去抢占优先级什么是子优先级)
    - [3.中断优先级配置方式](#3-中断优先级配置方式)
    - [4.freertos中对中断优先级的管理](#4-freertos中对中断优先级的管理)
  - [三.中断相关寄存器](#三中断相关寄存器)
    - [1.系统中断优先级配置寄存器](#1-系统中断优先级配置寄存器)
    - [2.FreeRTOS如何配置PendSV和Systick中断优先级](#2-freertos如何配置pendsv和systick中断优先级)
    - [3.为什么将PendSV和SysTick设置最低优先级](#3-为什么将pendsv和systick设置最低优先级)
    - [4.中断屏蔽寄存器](#4-中断屏蔽寄存器)
    - [5.BASEPRI中断屏蔽寄存器](#5-basepri中断屏蔽寄存器)
    - [6.freertos的关闭中断程序](#6-freertos的关闭中断程序)
    - [7.freertos的开中断程序](#7-freertos的开中断程序)
    - [8.中断服务函数调用FreeRTOS的API函数需注意](#8-中断服务函数调用freertos的api函数需注意)
</details>

<details>
<summary>6.freertos临界段代码保护</summary>

- [6.freertos临界段代码保护](#6-freertos临界段代码保护)
  - [1.什么是临界段](#1-什么是临界段)
  - [2.适用什么场合](#2-适用什么场合)
  - [3.什么可以打断当前程序的运行](#3-什么可以打断当前程序的运行)
  - [4.临界段代码保护函数](#4-临界段代码保护函数)
  - [5.临界段代码保护函数使用特点](#5-临界段代码保护函数使用特点)
</details>

<details>
<summary>7.任务调度器挂起和恢复函数</summary>

- [7.任务调度器挂起和恢复函数](#7-任务调度器挂起和恢复函数)
  - [1.任务调度器挂起和恢复函数](#1-任务调度器挂起和恢复函数)
  - [2.任务调度器挂起和恢复的特点](#2-任务调度器挂起和恢复的特点)
  - [3.挂起任务调度器vTaskSuspendAll](#3-挂起任务调度器vtasksuspendall)
  - [4.恢复任务调度器xTaskResumeAll](#4-恢复任务调度器xtaskresumeall)
</details>

<details>
<summary>8.freertos的列表和列表项</summary>

- [8.freertos的列表和列表项](#8-freertos的列表和列表项)
  - [一.列表和列表项的简介](#一列表和列表项的简介)
    - [1.什么是列表](#1-什么是列表)
    - [2.什么是列表项](#2-什么是列表项)
    - [3.列表和列表项的关系](#3-列表和列表项的关系)
    - [4.列表链表和数组的区别](#4-列表链表和数组的区别)
    - [5.OS中为什么使用列表](#5-os中为什么使用列表)
    - [6.列表结构体介绍](#6-列表结构体介绍)
    - [7.列表项结构体介绍](#7-列表项结构体介绍)
    - [8.迷你列表项](#8-迷你列表项)
    - [9.列表和列表项关系事例](#9-列表和列表项关系事例)
  - [二.列表相关的API函数介绍](#二列表相关的api函数介绍)
    - [1.列表API函数](#1-列表api函数)
    - [2.初始化列表函数vListInitialise](#2-初始化列表函数vlistinitialise)
    - [3.初始化列表项函数vListInitialiseItem](#3-初始化列表项函数vlistinitialiseitem)
    - [4.列表插入列表项函数vListInsert](#4-列表插入列表项函数vlistinsert)
    - [5.列表末尾插入列表项vListInsertEnd](#5-列表末尾插入列表项vlistinsertend)
    - [6.列表项移除函数uxListRemove](#6-列表项移除函数uxlistremove)
</details>

<details>
<summary>9.freertos任务调度</summary>

- [9.freertos任务调度](#9-freertos任务调度)
  - [一.开启任务调度器熟悉](#一开启任务调度器熟悉)
    - [1.开启任务调度器函数vTaskStartScheduler](#1-开启任务调度器函数vtaskstartscheduler)
    - [2.配置硬件架构及启动第一个任务函数xPortStartScheduler](#2-配置硬件架构及启动第一个任务函数xportstartscheduler)
    - [3.SysTick滴答定时器](#3-systick滴答定时器)
    - [4.堆和栈的地址生长方向](#4-堆和栈的地址生长方向)
    - [5.压栈和出栈的地址增长方向](#5-压栈和出栈的地址增长方向)
    - [6.知识补充](#6-知识补充)
  - [二.启动第一个任务熟悉](#二启动第一个任务熟悉)
    - [1.启动第一个任务涉及的关键函数](#1-启动第一个任务涉及的关键函数)
    - [2.想象一下应该如何启动第一个任务](#2-想象一下应该如何启动第一个任务)
    - [3.prvStartFirstTask 介绍](#3-prvstartfirsttask-介绍)
    - [4.什么是MSP指针](#4-什么是msp指针)
    - [5.为什么汇编代码要PRESERVE8八字节对齐](#5-为什么汇编代码要preserve8八字节对齐)
    - [6.prvStartFirstTask为什么要操作0XE00ED08](#6-prvstartfirsttask为什么要操作0xe00ed08)
    - [7.vPortSVCHandle介绍](#7-vportsvchandle介绍)
    - [8.出栈压栈汇编指令详解](#8-出栈压栈汇编指令详解)
  - [三.任务切换掌握](#三任务切换掌握)
    - [1.任务切换的本质](#1-任务切换的本质)
    - [2.任务切换过程](#2-任务切换过程)
    - [3.PendSV中断是如何触发的](#3-pendsv中断是如何触发的)
    - [4.在PendSV中断中PSP和MSP](#4-在pendsv中断中psp和msp)
    - [5.查找最高优先级任务](#5-查找最高优先级任务)
    - [6.前导置零指令](#6-前导置零指令)
</details>

<details>
<summary>10.FreeRTOS时间片轮询</summary>

- [10.FreeRTOS时间片轮询](#10-freertos时间片轮询)
  - [一.时间片轮询简介](#一时间片轮询简介)
</details>



## FreeRTOS

## 概述

随着产品实现的功能越来越多，单纯的裸机系统已经不能完美的解决问题了，反而会使程序边的更加复杂，如果想降低编程的难度，我们可以考虑引入RTOS实现多任务管理。

FreeRTOS由美国的Richard Barry于2003年发布，Richard Barry是FreeRTOS的拥有者和维护者，在过去的十多年 中FreeRTOS历经了9个版本，与众多半导体厂商合作密切，累计开发者数百万，是目前市场占有率最高的RTOS。

FreeRTOS是一款“开源免费”的实时操作系统，遵循的是GPLv2+的许可协议。这里说到的开源，指的是你可以免费得 获取到FreeRTOS的源代码，且当你的产品使用了FreeRTOS且没有修改FreeRTOS内核源码的时候，你的产品的全部代 码都可以闭源，不用开源，但是当你修改了FreeRTOS内核源码的时候，就必须将修改的这部分开源，反馈给社区， 其他应用部分不用开源。免费的意思是无论你是个人还是公司，都可以免费地使用，不需要掏一分钱。

## 1.基础知识

### 一.任务调度器简述

#### 1.什么是任务调度器

任务调度器是 FreeRTOS 的“大脑”，确保任务按照设计的要求以正确的顺序和时机执行。

#### 2.freertos的调度方式

- **抢占式调度** ：主要是针对优先级不同的任务，每一个任务都有一个任务优先级，优先级高的任务可以抢占低优先级的任务的CPU使用权。
- **时间片调度** ：主要针对相同优先级的任务，当多个任务的优先级相同时，任务调度器会在每个时钟节拍到来的时候切换任务。
- **协程式调度** ：其实就是轮询，当前执行任务将会一直运行，同时高优先级的任务不会抢占低优先级任务。FreeRTOS现在虽然还在支持，但官方已经明确表示不再更新协程式调度。

#### 3.抢占式调度过程

![抢占式调度](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/抢占式调度.png)

运行过程如下：

1. 首先Task1在运行中，在这个过程中Task2就绪了，在抢占式调度器的作用下，Task2会抢占Task1的运行。
2. Task2运行过程中，Task3就绪了，在抢占式调度器的作用下Task3会抢占Task2的运行。
3. Task3运行过程中，Task3阻塞了(系统延时或者等待信号等)，此时就绪中，优先级最高的任务Task2执行。
4. Task3阻塞解除了(延时到了或者接收到信号量)，此时Task3恢复到就绪态中，抢占Task2的运行。

总结：

1. 高优先级任务，优先执行。
2. 高优先级任务不停止，低优先级任务无法执行。
3. 被抢占的任务将会进去就绪态。

#### 4.时间片是什么？

同等优先级任务轮流享有相同的CPU时间(可设置)，叫做时间片，在FreeRTOS中，一个时间片等于SysTick中断周期。

#### 5.时间片调度过程

![时间片调度](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/时间片调度.png)

运行过程如下：

1. 首先Task1运行完一个时间片后，切换至Task2运行。
2. Task2运行完一个时间片后，切换至Task3运行。
3. Task3运行过程中(还不到一个时间片)，Task3阻塞了(系统延时或者等待信号量等)，此时直接切换到下一个任务Task1执行。
4. Task1运行完一个时间片后，切换至Task2运行。

总结：

1. 同等优先级任务，轮流执行。
2. 一个时间片大小，取决滴答定时器中断周期。
3. 没有用完的时间片不会再使用，任务Task3下次得到执行时间还是按照一个时间片的时钟节拍运行。

### 二.任务状态

#### 1.freertos的任务状态

FreeRTOS中任务存在4种状态：

- **运行态** ：正在执行的任务，该任务就处于运行状态(注意：在STM32中，同一时间仅一个任务处于运行态)。
- **就绪态** ：如果该任务已经能够被执行，但当前还未被执行，那么该任务处于就绪态。
- **阻塞态** ：如果一个任务因为延时或者等待外部事件发生，那么这个任务就处于阻塞态。
- **挂起态** ：类似于暂停，调用函数vTaskSuspend()进入挂起态，需要调用解挂函数vTaskResume()才可以进入就绪态。

#### 2.四种状态之间的转换关系

![任务状态转换关系](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/任务状态转换关系.png)

1. 仅就绪态可以转换成运行态。
2. 其他状态的任务想要运行，必须先转换成就绪态。

#### 3.任务状态列表

任务的四种状态中，除了运行态，其他三种任务状态的任务都有其对应的任务状态列表。

- **就绪列表** ： pxReadyTasksLists[x]，其中x代表人物优先级数值。
- **阻塞列表** ：pxDelayedTaskList
- **挂起列表** ：xSuspendedTaskList

## 2.freertos系统配置文件详解

freertosconfig.h配置文件的作用

对FreeRTOS进行功能配置和裁剪，以及API函数的使能。

学习途径

1. [官方的在线文档]([https://www.freertos.org/a00110.html](https://gitee.com/link?target=https%3A%2F%2Fwww.freertos.org%2Fa00110.html) )中有详细说的说明。
2. [正点原子《FreeRTOS开发指南》](https://gitcode.com/Open-source-documentation-tutorial/f681b/?utm_source=document_gitcode&index=top&type=card&&isLogin=1)第三章的内容 --- FreeRTOS系统配置。

配置文件中相关宏的分类

相关宏大致可以分为三类。

- **‘INCLUDE’开头** --- 配置FreeRTOS中可选的API函数。
- **’config‘开头** --- 完成FreeRTOS的功能配置和裁剪(如调度方式、使能信号量功能等)。
- **其他配置** --- PendSV宏定义、SVC宏定义。

## 3.任务的创建和删除

### 一.任务创建和删除API函数

#### 1.任务创建和删除的本质

任务创建和删除的本质就是调用FreeRTOS的API函数。

![任务创建和删除API函数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/任务创建和删除API函数.png)

#### 2.任务动态创建和静态创建的区别

- **动态创建任务** ：任务的任务控制块以及任务的栈空间所需的内存，均由FreeRTOS从FreeRTOS管理的堆中分配。
- **静态创建任务** ：任务的任务控制块以及任务的栈空间所需的内存，需要用户分配提供。

#### 3.任务控制块结构体成员介绍

![任务控制块结构体成员](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/任务控制块结构体成员.png)

1. 任务栈栈顶，在任务切换时的任务上下文保存、任务恢复息息相关。
2. 每个任务都有属于自己的任务控制块，类似身份证。

#### 4.什么是临界保护区

临界区保护，保护那些不想被打断的程序段，关闭freertos所管理的中断，中断无法打断，滴答中断和PendSV中断无法进行不能实现任务调度 。

#### 5.动态创建的优点

动态创建使用起来相对简单。在实际的应用中，动态方式创建任务是比较常用的，除非有特殊的需求，一般都会使用动态方式创建任务 。

#### 6.静态创建的优点

静态创建可将任务堆栈放置在特定的内存位置，并且无需关心对内存分配失败的处理 。

创建任务时，任务堆栈所存的内容

![任务创建时任务堆栈所存内容](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/任务创建时任务堆栈所存内容.png)

1. 寄存器下PSR被初始为0x01000000，其中bit24被置1，表示使用Thumb指令。
2. 寄存器PC被初始化为任务函数指针(任务A，即我们写的任务函数的地址)，这样当某次任务切换后，任务A获得CPU控制权，任务函数(任务A)被出栈到PC寄存器，之后会执行任务A的代码。
3. LR寄存器初始化为函数指针prvTaskExitError，这个函数是FreeRTOS提供的，是一个出错处理函数。
4. 子函数的调用通过寄存器R0~R3传递参数，创建任务时，我们传入的参数被保存到R0中，用来向任务传递参数。

### 二.任务的创建（动态）

#### 1.动态函数的创建

![动态任务创建函数参数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/动态任务创建函数参数.png)

函数返回值

![动态任务创建函数返回值](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/动态任务创建函数返回值.png)

#### 2.什么是句柄？（加）

你创建了一个任务，它就像是一个员工。

任务句柄（`TaskHandle_t`）就是这个任务的 **“身份证”**，让你以后可以随时找到这个任务、控制它，比如暂停、删除、调整优先级等。

有了任务名字，为什么还需要句柄呢？

1.任务名字是可以重复的

```
xTaskCreate(taskFunc, "Worker", 1000, NULL, 1, &worker1Handle);
xTaskCreate(taskFunc, "Worker", 1000, NULL, 1, &worker2Handle);
```

2.通过任务名查找效率低

3.再freertos中有些API不支持任务名字

| **场景**                     | **使用任务名字** | **使用任务句柄** |
| ---------------------------- | ---------------- | ---------------- |
| **调试**（查看任务列表）     | ✅                | ❌                |
| **日志记录**（打印任务信息） | ✅                | ❌                |
| **删除任务**                 | ❌                | ✅                |
| **挂起/恢复任务**            | ❌                | ✅                |
| **修改任务优先级**           | ❌                | ✅                |

#### 3.实现动态创建任务流程

只需要三步

1. 将FreeRTOSConfig.h文件中宏configSUPPORT_DYNAMIC_ALLOCATION配置为1。
2. 定义函数入口参数。
3. 编写任务函数。

动态任务创建函数创建的任务会立刻进入就绪态，由任务调度器调度运行。

#### 4.动态任务创建函数内部实现简述

1. 申请堆栈内存&任务控制块内存。
2. TCB结构体(任务控制块)成员赋值。
3. 添加新任务到就绪列表中。

### 三.任务的创建（静态）

#### 1.静态函数的创建

![静态任务创建函数参数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/静态任务创建函数参数.png)

函数返回值

![静态任务创建函数返回值](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/静态任务创建函数返回值.png)

#### 2.实现静态创建任务流程

1. 将FreeRTOSConfig.h文件中宏configSUPPORT_STATIC_ALLOCATION配置为1。
2. 定义空闲任务&定时器任务的任务堆栈以及TCB。
3. 实现两个接口函数(vAppLicationGetldleTaskMemory() 空闲任务接口函数和vApplicationGetTimerTaskMemory()定时器任务接口函数)。
4. 定义函数入口参数。
5. 编写任务函数。

静态任务创建函数创建的任务会立刻进入就绪态，由任务调度器调度运行。

#### 3.静态任务创建函数内部实现简述

1. TCB结构体成员赋值。
2. 添加新任务到就绪列表中。

### 四.任务的删除

#### 1.任务删除函数

![任务删除函数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/任务删除函数.png)

任务删除函数用于删除已经被创建的任务，被删除的任务将从就绪任务列表、阻塞任务列表、挂起任务列表和事件列表中移除。

1. 当传入的参数为NULL，则代表删除任务自身(当前正在运行的任务)。
2. 空闲任务会负责释放被删除任务中由系统分配的内存，但是由用户在任务删除前申请的内存空间，必须用户在任务被删除前提前释放，否则将会导致内存泄漏。

#### 2.删除任务流程

1. 使用删除任务函数，将FreeRTOSConfig.h文件中宏INCLUDE_vTaskDelete配置为1。
2. 入口参数输入需要删除的任务句柄(NULL代表删除本身)。

#### 3.删除任务函数内部实现简述

1. 获取所要删除的任务控制块 --- 通过传入的任务句柄，判断所需要删除哪个任务，NULL代表删除自身。
2. 将被删除任务移除所在列表 --- 将该任务所在列表中移除，包括：就绪、阻塞、挂起、事件等列表。
3. 判断所需要删除的任务
   - 删除任务自身，需要先添加到等待删除列表，内存释放将在空闲任务执行。
   - 删除其他任务，释放内存，任务数量。
4. 更新下个任务的阻塞时间 --- 更新下一个任务的阻塞超时时间，以防止被删除的任务就是下一个阻塞超时的任务。

## 4.任务的挂起和恢复

### 一.任务的挂起和恢复介绍

任务挂起与恢复的API函数

![任务挂起和恢复函数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/任务挂起和恢复函数.png)

- **挂起**：挂起任务类似暂停，可恢复；删除任务，无法恢复，类似“人死两清”。
- **恢复**：恢复被挂起的任务。
- **“FromISR”**：带有FromISR后缀是在中断函数中专用的API函数。

### 二.任务的挂起

#### 1.挂起函数介绍

![任务挂起函数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/任务挂起函数.png)

任务挂起函数用于挂起任务，使用时需要将FreeRTOSConfig.h文件中宏INCLUDE_vTaskSuspend配置为1。

无论优先级如何，被挂起的任务都将不再被执行，直到任务被恢复。

注意：当传入的参数是NULL，则代表挂起任务自身(当前正在运行的任务)。

#### 2.任务挂起函数内部实现

1. 获取所要挂起任务的控制块。（根据任务句柄获取任务控制块，如果任务句柄为NULL，表示挂起任务本身。）
2. 移除所在列表。（将要挂起的任务从相应的状态列表和事件列表中移除。）
3. 插入挂起任务列表。（将待挂起任务的任务状态列表项插入到挂起状态列表末尾。）
4. 判断任务调度器是否运行。（判断任务调度器是否运行，在运行，更新下一次阻塞时间，防止被挂起任务为下一个阻塞超时任务。）
5. 判断待挂起任务是否为当前任务。（如果挂起的是任务自身，且调度器正在运行，需要进行一次任务切换；调度器没有运行，判断挂起任务数是否等于任务总数，是：当前控制块赋值为NULL，否：寻找下一个最高优先级任务。）

### 三.任务的恢复

#### 1.任务恢复函数介绍（任务中）

![任务恢复函数(任务中使用)](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/任务恢复函数(任务中使用).png)

使用该函数时需要将FreeRTOSConfig.h文件中宏INCLUDE_vTaskSuspend配置为1。

注意：任务无论被vTaskSuspend()挂起多少次，只需在任务中调用vTaskResume()恢复一次就能继续运行，且被恢复的任务会进入就绪态。

#### 2.任务回复函数的实现（任务中）

1. 恢复任务不能是正在运行任务。
2. 判断任务是否子啊挂起列表中。（是：就会将该任务在挂起列表中移除，将该任务添加到就绪列表中。）
3. 判断恢复任务优先级。（判断恢复的任务优先级是否大于当前正在运行的任务，是的话，执行任务切换。）

#### 3.任务恢复函数介绍（中断中）

![任务恢复函数(中断中使用)](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/任务恢复函数(中断中使用).png)

使用该函数注意要将FreeRTOSConfig.h文件中宏INCLUDE_vTaskSuspend和INCLUDE_xTaskResumeFromISR配置为1。

该函数专用于中断服务函数中，用于解挂被挂起任务。

注意：中断服务程序中要调用freeRTOS的API函数则中断优先级不能高于FreeRTOS所管理的最高优先级。

#### 4.任务恢复函数内部实现（中断中）

1. 关闭freertos可管理中断，防止被其他的中断打断，并返回关闭前basepri寄存器的值。
2. 判断是否有挂起任务。
3. 将前面保存的basepri的值，恢复回来。
4. 返回xYieldRequired的值 用于决定是否需要进行任务切换。

## 5.中断管理

### 一.中断介绍

#### 1.什么是中断

让CPU打断正常运行的程序，转而去处理紧急的事件(程序)，就叫中断。

![中断举例](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/中断举例.png)

#### 2.中断执行机制

1. **中断请求** ：外设产生中断请求(如：GPIO外部中断、定时器中断...)。
2. **响应中断** ：CPU停止执行当前程序，转而执行中断处理程序(ISR)。
3. **退出中断** ：执行完毕，返回被打断的程序处，继续往下执行。

### 二.中断优先级分组设置

#### 1.中断优先级分组介绍

ARM Cortex-M 使用了8位宽的寄存器来配置中断的优先等级，这个寄存器就是中断优先级配置寄存器。但STM32只用了中断优先级配置寄存器的高4位[7:4]，所以STM32提供了最大16级(0~15)的中断优先等级。

![优先级配置寄存器](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/优先级配置寄存器.png)

#### 2.什么是去抢占优先级，什么是子优先级

- **抢占优先级** ：抢占优先级的中断可以打断正在执行但抢占优先级低的中断。
- **子优先级** ：当同时发生具有相同抢占优先级的两个中断时，子优先级数小的优先执行。

注意 ：中断优先级(抢占优先级和子优先级)数值越小，优先级越高。

#### 3.中断优先级配置方式

一共有5种配置方式，对应着中断优先级分组的5个组。

![中断优先级分配方式](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/中断优先级分配方式.png)

FreeRTOS中为了方便管理，采用第4号(NVIC_PriorityGroup_4)分配方式。

#### 4.freertos中对中断优先级的管理

1. 低于configMAX_SYSCALL_INTERRUPT_PRIORITY优先级的中断才允许调用FreeRTOS的API函数。
2. 建议将所有优先级位指定为抢占优先级位，方便FreeRTOS管理。
3. 中断优先级数值越小越优先，任务优先级数值越大越优先。

![中断和任务优先级的不同](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/中断和任务优先级的不同.png)

### 三.中断相关寄存器

#### 1.系统中断优先级配置寄存器

三个系统中断优先级配置寄存器，分别为 SHPR1、 SHPR2、 SHPR3 。

- SHPR1寄存器地址：0xE000ED18~0xE000ED1B
- SHPR2寄存器地址：0xE000ED1C~0xE000ED1F
- SHPR3寄存器地址：0xE000ED20~0xE000ED23

![系统中断配置寄存器](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/系统中断配置寄存器.png)

FreeRTOS主要是使用SHPR3寄存器对**PendSV**和**Systick**中断优先级进行设置(设置为最低优先级)

#### 2.FreeRTOS如何配置PendSV和Systick中断优先级

![PendSV和Systick中断优先级1](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/PendSV和Systick中断优先级1.png)

![PendSV和Systick中断优先级2](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/PendSV和Systick中断优先级2.png)

![PendSV和Systick中断优先级3](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/PendSV和Systick中断优先级3.png)

![PendSV和Systick中断优先级4](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/PendSV和Systick中断优先级4.png)

在FreeRTOS系统中PendSV和SysTick设置最低优先级。

#### 3.为什么将PendSV和SysTick设置最低优先级

保证系统任务切换不会阻塞系统其他中断的响应。

#### 4.中断屏蔽寄存器

三个中断屏蔽寄存器，分别为PRIMASK、FAULTMASK和BASEPRI 。

![中断屏蔽寄存器](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/中断屏蔽寄存器.png)

FreeRTOS所使用的中断管理就是利用的**BASEPRI**这个寄存器。

#### 5.BASEPRI中断屏蔽寄存器

BASEPRI：屏蔽优先级低于某一个阈值的中断，当设置为0时，则不关闭任何中断。

比如： BASEPRI设置为0x50，代表中断优先级在5~15内的均被屏蔽，0~4的中断优先级正常执行

![BASEPRI中断屏蔽寄存器事例](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/BASEPRI中断屏蔽寄存器事例.png)

#### 6.freertos的关闭中断程序

![关中断程序](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/关中断程序.png)

#### 7.freertos的开中断程序

![开中断程序](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/开中断程序.png)

#### 8.中断服务函数调用FreeRTOS的API函数需注意

1. 中断服务函数的优先级需在FreeRTOS所管理的范围内。
2. 在中断服务函数里边需调用FreeRTOS的API函数，必须使用带“FromISR”后缀的函数。

## 6.freertos临界段代码保护

#### 1.什么是临界段

临界段代码也叫临界区，是指那些必须完整运行，不能被打断的代码片段。运行时临界段代码时需要关闭中断，当处理完临界段代码以后再打开中断。

#### 2.适用什么场合

- **外设** ：需要严格按照时序初始化的外设，如IIC、SPI等。
- **系统** ：系统自身需求，如任务切换过程等。
- **用户** ：用户需求，如我们写的任务创建任务。

#### 3.什么可以打断当前程序的运行

中断、任务调度。

#### 4.临界段代码保护函数

![临界段保护函数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/临界段保护函数.png)

任务级临界区调用格式示例：

```c
taskENTER_CRITICAL();
{
	... ... /* 临界区 */
}
taskENTER_CRITICAL()
```

中断级临界区调用格式示例：

```c
uint32_t save_status;
save_status = taskENTER_CRITICAL_FROM_ISR();
{
	... ... /* 临界区 */
}
taskENTER_CRITICAL_FROM_ISR(save_status);
```

#### 5.临界段代码保护函数使用特点

1. 成对使用。
2. 支持嵌套。
3. 尽量保持临界段耗时短。

## 7.任务调度器挂起和恢复函数

#### 1.任务调度器挂起和恢复函数

![任务调度器挂起和恢复函数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/任务调度器挂起和恢复函数.png)

使用格式示范：

```c
vTaskSuspendAll();
{
	... ... /* 内容 */
}
vTaskSuspendAll();
```

#### 2.任务调度器挂起和恢复的特点

1. 与临界区不一样的是，挂起任务调度器，未关闭中断。
2. 它仅仅是防止任务之间的资源争夺，中断照样可以直接响应。
3. 挂起任务调度器的方式，适用于临界区位于任务与任务之间；既不用去延时中断响应，又可以做到临界区的安全。

#### 3.挂起任务调度器：vTaskSuspendAll()

调用一次挂起调度器，该变量uxSchedulerSuspended就加一 ，变量uxSchedulerSuspended的值，将会影响Systick触发PendSV中断，即影响任务调度。

#### 4.恢复任务调度器：xTaskResumeAll()

调用一次恢复调度器，该变量uxSchedulerSuspended就减一 ，如果uxSchedulerSuspended等于0，则允许调度 。

1. 当任务数量大于0时，恢复调度器才有意义，如果没有一个已创建的任务就无意义。
2. 移除等待就绪列表中的列表项,恢复至就绪列表,直到xPendingReadyList列表为空。
3. 如果恢复的任务优先级比当前正在执行任务优先级更高，则将xYieldPending赋值为pdTRUE,表示需要进行一次任务切换。
4. 在调度器被挂起的期间内,是否有丢失未处理的滴答数。 xPendedCounts是丢失的滴答数，有则调用xTasklncrementTickf() 补齐弄失的滴答数。
5. 判断是否允许任务切换。
6. 返回任务是否已经切换；已经切换返回pdTRUE；反之返回pdFALSE。

## 8.freertos的列表和列表项

### 一.列表和列表项的简介

#### 1.什么是列表

列表是FreeRTOS中的一个数据结构，概念上和链表有点类似，列表被用来跟踪FreeRTOS中的任务。

#### 2.什么是列表项

列表项就是存放在列表中的项目。

#### 3.列表和列表项的关系

列表相当于链表，列表项相当于节点，FreeRTOS中的列表是一个双向环形链表。

![列表和列表项关系](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/列表和列表项关系.png)

#### 4.列表(链表)和数组的区别

- **列表的特点**：列表项间的地址非连续的，是人为的连接到一起的。列表项的数目是由后期添加或删除的个数决定的，随时可以改变。
- **数组的特点**：数组成员地址是连续的，数组在最初确定了成员数量后，后期将无法改变。

#### 5.OS中为什么使用列表

在OS中任务的数量是不确定的，并且任务状态是会发生改变的，所以非常适用列表(链表)这种数据结构。

#### 6.列表结构体介绍

有关列表的东西均在文件list.c和list.h中，以下是列表结构体：

![列表结构体](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/列表结构体.png)

1. 在结构体中，包含两个宏(listFIRST_LIST_INTEGRITY_CHECK_VALUE 和 listSECOND_LIST_INTEGRITY_CHECK_VALUE)，这两个宏是确定的已知常量，FreeRTOS通过检查这两个常量的值，来判断列表的数据在程序运行过程中，是否遭到破坏，该功能一般用于测试，默认是不开启的(我们一般不用去理会)。
2. 成员uxNumberOfltems，用于记录列表中的列表项的个数(不包括xListEnd)。
3. 成员pxIndex用于指向列表中的某个列表项，一般用于遍历列表中的所有列表项。
4. 成员变量xListEnd是一个迷你列表项，排在最末尾。

列表结构示意图：

![列表结构示意图](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/列表结构示意图.png)

#### 7.列表项结构体介绍

列表项是列表中用于存放数据的地方，在list.h文件中，列表项的相关结构体定义：

![列表项结构体](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/列表项结构体.png)

1. 成员变量xItemValue为列表项的值，这个值多用于按升序对列表中的列表项进行排序。
2. 成员变量pxNext和pxPrevious分别用于指向列表中列表项的下一个列表项和上一个列表项。
3. 成员变量pxOwner用于指向包含列表项的对象(通常是任务控制块)。
4. 成员变量pxContainer用于执行列表项所在列表。

列表项结构体示意图：

![列表项结构体示意图](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/列表项结构体示意图.png)

#### 8.迷你列表项

迷你列表项也是列表项，但迷你列表项仅用于标记列表的末尾和挂载其他插入列表中的列表项。

![迷你列表项](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/迷你列表项.png)

1. 成员变量xItemValue为列表项的值，这个值多用于按升序对列表中的列表项进行排序。
2. 成员变量pxNext和pxPrevious分别用于指向列表中列表项的下一个列表项和上一个列表项。
3. 迷你列表项只用于标记列表的末尾和挂载其他插入列表中的列表项，因此不需要成员变量pxOwner和pxContainer，以节省内存开销。

迷你列表项示意图：

![迷你列表项示意图](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/迷你列表项示意图.png)

#### 9.列表和列表项关系事例

列表初始状态：

![列表初始化状态](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/列表初始化状态.png)

列表插入两个列表项：

![列表插入列表项](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/列表插入列表项.png)

列表当前状态：

![当前列表简图](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/当前列表简图.png)

### 二.列表相关的API函数介绍

#### 1.列表API函数

![列表函数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/列表函数.png)

#### 2.初始化列表函数vListInitialise()

![初始化列表函数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/初始化列表函数.png)

函数参数：

![初始化列表函数参数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/初始化列表函数参数.png)

列表初始化后示意图：

![初始化列表函数使用后](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/初始化列表函数使用后.png)

#### 3.初始化列表项函数vListInitialiseItem()

![初始化列表项函数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/初始化列表项函数.png)

函数参数：

![初始化列表项函数参数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/初始化列表项函数参数.png)

列表初始化后示意图：

![初始化列表项函数使用后](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/初始化列表项函数使用后.png)

#### 4.列表插入列表项函数vListInsert()

此函数用于将待插入列表的列表项按照列表项值升序进行排序，有序地插入到列表中 。

![列表插入列表项函数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/列表插入列表项函数.png)

函数参数：

![列表插入列表项函数参数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/列表插入列表项函数参数.png)

函数vListInsert()，是将插入列表的列表项按照列表项值升序进行排列，有序地插入到列表中。

#### 5.列表末尾插入列表项vListInsertEnd()

此函数用于将待插入列表的列表项插入到列表 pxIndex 指针指向的列表项前面，是一种无序的插入方法。

![列表末尾插入列表项函数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/列表末尾插入列表项函数.png)

函数参数：

![列表末尾插入列表项函数参数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/列表末尾插入列表项函数参数.png)

#### 6.列表项移除函数uxListRemove()

此函数用于将列表项从列表项所在列表中移除。

![列表项移除函数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/列表项移除函数.png)

函数参数：

![列表项移除函数参数](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/列表项移除函数参数.png)

函数返回值：

![列表项移除函数返回值](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/列表项移除函数返回值.png)

## 9.freertos任务调度

### 一.开启任务调度器(熟悉)

#### 1.开启任务调度器函数vTaskStartScheduler()

答：作用：启动任务调度器，任务调度器启动后，FreeRTOS便会开始进行任务调度。

该函数内部实现，如下：

1. 创建空闲任务。
2. 如果使能软件定时器，则创建定时器任务。
3. 关闭中断，防止调度器开启之前或过程中，受到中断干扰，会在运行第一个任务时打开中断。
4. 初始化全局变量，并将任务调度器的运行标志设置为已运行。
5. 初始化任务运行时间统计功能的时基定时器。
6. 调用函数xPortStartScheduler()。

#### 2.配置硬件架构及启动第一个任务函数xPortStartScheduler()

答：作用：该函数用于完成启动任务调度器中与硬件架构相关配置部分，以及启动第一个任务。

该函数内部实现，如下：

1. 检测用户在FreeRTOSConfig.h文件中对中断的配置是否有误。
2. 配置PendSV和SysTick的中断优先为最低优先级。
3. 调用函数vPortSetupTimerInterrupt()配置SysTick。
4. 初始化临界区嵌套计数器为0。
5. 调用函数prvEnableVFP()使能FPU。
6. 调用函数prvStartFirstTask()启动第一个任务。

#### 3.SysTick滴答定时器

答：

![滴答定时器](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/滴答定时器.png)

#### 4.堆和栈的地址生长方向

答：

**堆**  的生长方向向上，内存地址由低到高；

**栈**  的生长方向向下，内存地址由高到低。

#### 5.压栈和出栈的地址增长方向

答：堆栈的生长方向从最本质的理解是堆栈入栈方向是从高地址向地址还是低地址向高地址。

从高地址向低地址生长一般叫做向下生长，也叫作逆向生长。

从低地址向高地址生长一般叫做向上生长，也叫作正向生长。

一般来说堆栈是逆向生长的。

- 51单片机的堆栈生长方向为正向生长，因为执行PUSH指令时先将SP的值加1再将指定的8位数据单元的内容入栈。

- 80x86微机的堆栈生长方向为逆向生长，因为执行PUSH指令时先将SP的值减2再将指定的16位数据单元内容入栈。高字节放高地址，低字节放低地址（小端模式）。

- STM32的堆栈生长方向是逆向生长。

注意：C语言中传递参数，一般是从右向左入栈的，所以最左边的参数是最先出栈的。

#### 6.知识补充

1s =  1000ms

1ms = 1000us

1us = 1000ns

1Mhz = 1000 000hz

hz转换为s公式：1s =  1/(1hz)

例如：10hz等于0.1s (0.1s = 1 / 10hz)。

### 二、启动第一个任务(熟悉)

#### 1.启动第一个任务涉及的关键函数

答：

prvStartFirstTask()      /* 启动第一个任务 */

vPortSVCHandle()     /* SVC中断服务函数 */

#### 2.想象一下应该如何启动第一个任务 

答：假设我们要启动的第一个任务是任务A，那么就需要将任务A的寄存器值恢复到CPU寄存器中。任务A的寄存器值，在一开始创建任务时就已经保存在任务堆栈里边了！

注意：

1. 中断产生时，硬件自动将xPSR、PC(R15)、LR(R14)、R12、R3\~R0保存和恢复，而R4\~R11需要手动保存和恢复。
2. 进入中断后，硬件会强制使用MSP指针，此时LR(R14)的值将会被自动更新为特殊的EXC_RETURN。

#### 3.prvStartFirstTask() 介绍

答：用于初始化启动第一个任务前的环境，主要是重新设置MSP指针，并使能全局中断，最后触发SVC中断。

#### 4.什么是MSP指针

答：程序在运行过程中需要一定的栈空间来保存局部变量等信息。当有信息保存到栈中时，MCU会自动更新SP指针，ARM Cortex-M内核提供了两个栈空间：

- **主堆栈指针(MSP)**：它是给OS内核、异常服务程序以及所有需要特权访问的应用程序代码来使用的。
- **进程堆栈指针(PSP)**：用于常规的应用程序代码(不处于异常服务程序中时使用)。

在FreeRTOS中，中断使用MSP(主堆栈)，中断以为使用PSP(进程堆栈)。

注意：在RTOS中是使用双堆栈指针(即使用MSP和PSP)，但在裸机中是只使用MSP(主堆栈)。

#### 5.为什么汇编代码要PRESERVE8(八字节对齐)

答：因为栈在任何时候都是需要4字节对齐的，而在调用入口得8字节对齐，在C编程的时候，编译器会自动帮我们完成对齐操作，而汇编则需要手动对齐。

#### 6.prvStartFirstTask()为什么要操作0XE00ED08

答：因为需要从0XE000ED08获取向量表的偏移，为啥要获取向量表呢？因为向量表的第一个是MSP指针！获取MSP的初始值的思路是先根据向量表的位置寄存器VTOR(0XE000ED08)来获取向量表存储的地址，再根据向量表存储的地址，来访问第一个元素，也就是初始的MSP。

#### 7.vPortSVCHandle()介绍

答：当使能了全局中断，并且手动触发SVC中断后，就会进到SVC的中断服务函数中。

1. 通过pxCurrentTCB获取优先级最高的就绪态任务的任务栈地址，优先级最高的就绪态任务是系统将要运行的任务。
2. 通过任务的栈顶指针，将任务栈中的内容出栈到CPU寄存器中，任务栈中的内容在调用任务创建函数的时候，已初始化，然后设置PSP指针。
3. 通过往BASEPRI寄存器中写0，允许中断。
4. R14是链接寄存器LR，在ISR中(此刻我们在SVC的ISR中)，它记录了异常返回值EXC_RETURN，而EXC_RETURN只有6个合法的值(M4、M7)，如下表所示：![R14(LR链接寄存器)](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/R14(LR链接寄存器).png)

注意：

1. SVC中断只在启动第一次任务时会调用一次，以后均不调用。
2. R14(LR链接寄存器)，在异常处理期间(就是中断函数中)，LR存的是特殊的EXC_RETURN(异常返回)数值，该数值会在异常处理结束时触发异常返回；在普通函数中，LR用于函数或者子程序调用时返回地址的保存。

#### 8.出栈/压栈汇编指令详解

答：

1. **出栈（恢复现场）**，方向：从下往上（低地址往高地址）：假设r0地址为0x04汇编指令示例：

   ldmia r0!, {r4-r6}  /* 任务栈r0地址由低到高，将r0存储地址里面的内容手动加载到 CPU寄存器r4、r5、r6 */

   r0地址(0x04)内容加载到r4，此时地址r0 = r0+4 = 0x08

   r0地址(0x08)内容加载到r5，此时地址r0 = r0+4 = 0x0C

   r0地址(0x0C)内容加载到r6，此时地址r0 = r0+4 = 0x10

2. **压栈（保存现场）**，方向：从上往下（高地址往低地址）：假设r0地址为0x10汇编指令示例：

   stmdb r0!, {r4-r6} }  /* r0的存储地址由高到低递减，将r4、r5、r6里的内容存储到r0的任务栈里面。 */

   地址：r0 = r0-4 = 0x0C，将r6的内容（寄存器值）存放到r0所指向地址(0x0C)

   地址：r0 = r0-4 = 0x08，将r5的内容（寄存器值）存放到r0所指向地址(0x08)

   地址：r0 = r0-4 = 0x04，将r4的内容（寄存器值）存放到r0所指向地址(0x04)

### 三.任务切换(掌握)

#### 1.任务切换的本质

答：任务切换的本质就是CPU寄存器内容的切换。

假设当由任务A切换到任务B时，主要分为两步：

**第一步**：需暂停任务A的执行，并且将此时任务A寄存器保存到任务堆栈中，这个过程叫做保存现场；

**第二步**：将任务B的各个寄存器值(被存于任务堆栈中)恢复到CPU寄存器中，这个过程叫做恢复现场。

对任务A保存现场，对任务B恢复现场，这个过程称为：**上下文切换**。

#### 2.任务切换过程

答：![任务切换](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/任务切换.png)

注意：任务切换的过程在PendSV中断服务函数里边完成。

#### 3.PendSV中断是如何触发的

答：

1. 滴答定时器中断调用
2. 执行FreeRTOS提供的相关API函数：portYIELD()。

本质是通过中断控制和状态寄存器ICSR的bit28写入1挂起PendSV来启动PendSV中的。

![PendSV相关寄存器](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/PendSV相关寄存器.png)

上表摘取于《Cortex M3权威指南(中文)》第131页。

#### 4.在PendSV中断中PSP和MSP

答：在进入PendSV异常前的自动压栈使用的是进程堆栈(PSP)，正式进入到PendSV异常Handle后才自动改为主堆栈(MSP)，退出异常时切回PSP，并且从进程堆栈(PSP)上弹出数据(出栈)。

![进入PendSV时的PSP和MSP](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/进入PendSV时的PSP和MSP.png)

注意：在PendSV中断中时，CPU使用的是MSP，但我们要处理的是PSP的数据存取。

#### 5.查找最高优先级任务

答：

vTaskSwitchContext( )                                         /* 查找最高优先级任务 */

taskSELECT_HIGHEST_PRIORITY_TASK( )    /* 通过这个函数完成 */

#### 6.前导置零指令

答：

![前导置零指令](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/前导置0指令.png)

所谓的前导置零指令，大家可以简单理解为计算一个 32位数，出现第一个1前头部 0 的个数。

------

## 10.FreeRTOS时间片轮询

#### 一.时间片轮询简介

答：同等优先级任务轮流享有相同的CPU时间(可设置)，叫**时间片**，在FreeRTOS中，一个时间片就等于SysTick中断周期。

![时间片调度](https://raw.githubusercontent.com/Aloner63/mymm/typora/typora/freertos/时间片调度.png)

运行过程如下：

1. 首先Task1运行完一个时间片后，切换至Task2运行。
2. Task2运行完一个时间片后，切换至Task3运行。
3. Task3运行过程中(还不到一个时间片)，Task3阻塞了(系统延时或等待信号量等)，此时直接切换到下一个任务Task1.
4. Task1运行完一个时间片后，切换Task2运行。

总结：

1. 同等优先级任务，轮流执行，时间片流转。
2. 一个时间片大小，取决为滴答定时器中断频率。
3. 注意没有完成的时间片不会再使用，下次任务Task3得到执行还是按照一个时间片的时钟节拍执行。



