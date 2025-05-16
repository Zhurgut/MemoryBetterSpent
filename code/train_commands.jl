


begin 
    id = 0
    # collect_measurements(dense, 1024, 4, 1e-4, 1000, 50, 1, id=id)
    collect_measurements(vit_dense, 8*128, 6, 5e-4, 128, 30, 1, (patch_size=4, nr_heads=8),  max_bs=500, weight_decay=0.01, id=id)
end



begin
    id = 9
    # collect_measurements(dense, 1024, 4, 4e-4, 500, 3, id=9)
    collect_measurements(monarch, 1024, 4, [1e-2, 1e-3, 3e-3, 3e-4, 1e-4], 1000, 500, 1, [(nr_blocks=4,)], id=id)
    collect_measurements(monarch, 1024, 4, [1e-2, 1e-3, 3e-3, 3e-4, 1e-4], 1000, 500, 1, [(nr_blocks=8,)], id=id)
    collect_measurements(monarch, 1024, 4, [1e-2, 1e-3, 3e-3, 3e-4, 1e-4], 1000, 500, 1, [(nr_blocks=16,)], id=id)
    collect_measurements(monarch, 1024, 4, [1e-2, 1e-3, 3e-3, 3e-4, 1e-4], 1000, 500, 1, [(nr_blocks=32,)], id=id)

    collect_measurements(monarch, 1024, 4, [3e-4, 1e-4], 1000, 500, 1, (nr_blocks=4,), weight_decay=[1e-2, 1e-3, 1e-4], id=id)
    collect_measurements(monarch, 1024, 4, [6e-4, 3e-4], 1000, 500, 1, (nr_blocks=8,), weight_decay=[1e-2, 1e-3, 1e-4], id=id)
    collect_measurements(monarch, 1024, 4, [1e-3, 5e-4], 1000, 500, 1, (nr_blocks=16,), weight_decay=[1e-2, 1e-3, 1e-4], id=id)
    collect_measurements(monarch, 1024, 4, [2e-3, 4e-3], 1000, 500, 1, (nr_blocks=32,), weight_decay=[1e-2, 1e-3, 1e-4], id=id)
end

begin 
    id = 10
    collect_measurements(dense, 256,  4, [1e-2, 3e-3, 1e-3, 3e-4], 1000, 500, 1, id=id)
    collect_measurements(dense, 512,  4, [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], 1000, 500, 1, id=id)
    collect_measurements(dense, 1024, 4, [3e-3, 1e-3, 3e-4, 1e-4], 1000, 500, 1, id=id)
end

begin 
    id = 10
    collect_measurements(dense, 576,  4, [3e-3, 1e-3], 1000, 500, 1, id=id)
    collect_measurements(dense, 640,  4, [2e-3, 9e-4], 1000, 500, 1, id=id)
    collect_measurements(dense, 704,  4, [1e-3, 7e-4], 1000, 500, 1, id=id)
    collect_measurements(dense, 768,  4, [8e-4, 5e-4], 1000, 500, 1, id=id)
    collect_measurements(dense, 832,  4, [6e-4, 4e-4], 1000, 500, 1, id=id)
    collect_measurements(dense, 896,  4, [5e-4, 3e-4], 1000, 500, 1, id=id)
    collect_measurements(dense, 960,  4, [4e-4, 2e-4], 1000, 500, 1, id=id)
    collect_measurements(dense, 1024, 3, 3e-4, 1000, 500, 1, id=id)
end

begin
    id = 11
    collect_measurements(lowrank, 256,  4, [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], 1000, 500, 1, (rank=128,), id=id)
    collect_measurements(lowrank, 256,  4, [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], 1000, 500, 1, (rank=64,), id=id)
    collect_measurements(lowrank, 256,  4, [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], 1000, 500, 1, (rank=32,), id=id)
    collect_measurements(lowrank, 256,  4, [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], 1000, 500, 1, (rank=16,), id=id)
    collect_measurements(lowrank, 512,  4, [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], 1000, 500, 1, (rank=256,), id=id)
    collect_measurements(lowrank, 512,  4, [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], 1000, 500, 1, (rank=128,), id=id)
    collect_measurements(lowrank, 512,  4, [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], 1000, 500, 1, (rank=64,), id=id)
    collect_measurements(lowrank, 512,  4, [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], 1000, 500, 1, (rank=32,), id=id)
    collect_measurements(lowrank, 1024,  4, [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], 1000, 500, 1, (rank=512,), id=id)
    collect_measurements(lowrank, 1024,  4, [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], 1000, 500, 1, (rank=256,), id=id)
    collect_measurements(lowrank, 1024,  4, [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], 1000, 500, 1, (rank=128,), id=id)
    collect_measurements(lowrank, 1024,  4, [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], 1000, 500, 1, (rank=64,), id=id)

end

begin
    id = 12

    collect_measurements(monarch, 1280, [3, 4, 5], 1e-4, 1000, 500, 1, [(nr_blocks=4,)], id=id)
    collect_measurements(monarch, 1024, [4, 5, 6], [4e-4], 1000, 500, 1, [(nr_blocks=8,)], id=id)
    collect_measurements(monarch, 1024, [5, 6, 7, 8], [1e-3, 5e-4], 1000, 500, 1, [(nr_blocks=16,)], id=id)

end


begin
    id = 13
    ranks = [32, 64, 128, 256]  

    for rank in ranks
        collect_measurements(tt, 1296, 3, [1e-4, 1e-3, 1e-2], 1000, 500, 1, [(nr_cores=2, rank=rank)], id=id) # coresize: 36
        collect_measurements(tt, 1331, 3, [1e-4, 1e-3, 1e-2], 1000, 500, 1, [(nr_cores=3, rank=rank)], id=id) #           11
        collect_measurements(tt, 1296, 3, [1e-4, 1e-3, 1e-2], 1000, 500, 1, [(nr_cores=4, rank=rank)], id=id) #           6
        
        
    end

end

begin
    id = 14
    ranks = [16, 256, 8]  

    for i=1:3
        collect_measurements(tt, 1000, 4, 1e-3, 1000, 10, 1, [(nr_cores=3, rank=ranks[i])], id=id) #           10
    end

end

begin
    id = 15
    ranks = [16, 32, 64, 128, 192, 256]  
    lrs   = [3e-3, 2.5e-3, 2e-3, 1.5e-3, 1e-3, 8e-4]

    for i=1:6
        collect_measurements(tt, 1024, 4, lrs[i], 1000, 500, 1, [(nr_cores=2, rank=ranks[i])], id=id) # coresize: 32
    end

    for i=1:3
        collect_measurements(tt, 1000, 4, lrs[i], 1000, 500, 1, [(nr_cores=3, rank=ranks[i])], id=id) #           10
    end

    for i=1:6
        collect_measurements(tt, 1024, 4, lrs[i], 1000, 500, 1, [(nr_cores=5, rank=ranks[i])], id=id) # coresize: 2
    end

end



begin
    id = 16

    collect_measurements(monarch, 1024, 4, [6e-5,   1e-4   ,2e-4]   , 1000, 500, 1, (nr_blocks=2,), init_scale=[0.7, 1.0, 1.4, 2.0, 3.0], id=id) # 512
    collect_measurements(monarch, 1023, 4, [1e-4,   1.5e-4 ,2.5e-4] , 1000, 500, 1, (nr_blocks=3,), init_scale=[0.7, 1.0, 1.4, 2.0, 3.0], id=id) # 341
    collect_measurements(monarch, 1024, 4, [1e-4,   2e-4   ,3e-4]   , 1000, 500, 1, (nr_blocks=4,), init_scale=[0.7, 1.0, 1.4, 2.0, 3.0], id=id) # 256
    collect_measurements(monarch, 1025, 4, [1.5e-4, 2.5e-4 ,4e-4]   , 1000, 500, 1, (nr_blocks=5,), init_scale=[0.7, 1.0, 1.4, 2.0, 3.0], id=id) # 205
    collect_measurements(monarch, 1026, 4, [1.7e-4, 3e-4   ,5e-4]   , 1000, 500, 1, (nr_blocks=6,), init_scale=[0.7, 1.0, 1.4, 2.0, 3.0], id=id) # 171
    collect_measurements(monarch, 1022, 4, [2e-4,   3.5e-4 ,6e-4]   , 1000, 500, 1, (nr_blocks=7,), init_scale=[0.7, 1.0, 1.4, 2.0, 3.0], id=id) # 146
    collect_measurements(monarch, 1024, 4, [2.5e-4, 4e-4   ,7e-4]   , 1000, 500, 1, (nr_blocks=8,), init_scale=[0.7, 1.0, 1.4, 2.0, 3.0], id=id) # 128

end


begin
    id = 20

    collect_measurements(btt, 1024, 4, 1e-3,   1000, 200, 1, (nr_cores=2, rank=1), init_scale=[0.7, 1.0, 1.4], weight_decay=[0.0, 0.01], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, 1e-3,   1000, 200, 1, (nr_cores=2, rank=2), init_scale=[0.7, 1.0, 1.4], weight_decay=[0.0, 0.01], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, 1e-3,   1000, 200, 1, (nr_cores=2, rank=4), init_scale=[0.7, 1.0, 1.4], weight_decay=[0.0, 0.01], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, 1e-3,   1000, 200, 1, (nr_cores=2, rank=8), init_scale=[0.7, 1.0, 1.4], weight_decay=[0.0, 0.01], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, 1e-3,   1000, 200, 1, (nr_cores=2, rank=12), init_scale=[0.7, 1.0, 1.4], weight_decay=[0.0, 0.01], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, 1e-3,   1000, 200, 1, (nr_cores=2, rank=16), init_scale=[0.7, 1.0, 1.4], weight_decay=[0.0, 0.01], id=id) # coresize: 32

    collect_measurements(btt, 1000, 4, 1e-3,   1000, 200, 1, (nr_cores=3, rank=1), init_scale=[0.7, 1.0, 1.4], weight_decay=[0.0, 0.01], id=id) # coresize: 32
    collect_measurements(btt, 1000, 4, 1e-3,   1000, 200, 1, (nr_cores=3, rank=2), init_scale=[0.7, 1.0, 1.4], weight_decay=[0.0, 0.01], id=id) # coresize: 32
    collect_measurements(btt, 1000, 4, 1e-3,   1000, 200, 1, (nr_cores=3, rank=4), init_scale=[0.7, 1.0, 1.4], weight_decay=[0.0, 0.01], id=id) # coresize: 32
    collect_measurements(btt, 1000, 4, 1e-3,   1000, 200, 1, (nr_cores=3, rank=8), init_scale=[0.7, 1.0, 1.4], weight_decay=[0.0, 0.01], id=id) # coresize: 32

    collect_measurements(btt, 1024, 4, 5e-3,   1000, 200, 1, (nr_cores=5, rank=1), init_scale=[0.7, 1.0, 1.4], weight_decay=[0.0, 0.01], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, 5e-3,   1000, 200, 1, (nr_cores=5, rank=2), init_scale=[0.7, 1.0, 1.4], weight_decay=[0.0, 0.01], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, 5e-3,   1000, 200, 1, (nr_cores=5, rank=4), init_scale=[0.7, 1.0, 1.4], weight_decay=[0.0, 0.01], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, 5e-3,   1000, 200, 1, (nr_cores=5, rank=8), init_scale=[0.7, 1.0, 1.4], weight_decay=[0.0, 0.01], id=id) # coresize: 32

end

begin
    id = 18
    ranks = [16, 32, 64, 128, 192, 256]  
    lrs   = [3e-3, 2.5e-3, 2e-3, 1.5e-3, 1e-3, 8e-4]


    collect_measurements(tt, 1024, 4, [8e-4, 1e-3, 1.5e-3], 1000, 500, 1, (nr_cores=2, rank=128), init_scale=[0.9, 1.0, 1.1], id=id) # coresize: 32
    collect_measurements(tt, 1024, 4, [6e-4, 8e-4, 1e-3],   1000, 500, 1, (nr_cores=2, rank=192), init_scale=[0.9, 1.0, 1.1], id=id) # coresize: 32
    collect_measurements(tt, 1024, 4, [5e-4, 8e-4, 1e-3],   1000, 500, 1, (nr_cores=2, rank=256), init_scale=[0.9, 1.0, 1.1], id=id) # coresize: 32
    collect_measurements(tt, 1024, 4, [3e-4, 6e-4, 9e-4],   1000, 500, 1, (nr_cores=2, rank=320), init_scale=[0.9, 1.0, 1.1], id=id) # coresize: 32
    collect_measurements(tt, 1024, 4, [2e-4, 4e-4, 7e-4],   1000, 500, 1, (nr_cores=2, rank=384), init_scale=[0.9, 1.0, 1.1], id=id) # coresize: 32

    collect_measurements(tt, 1000, 4, [7e-4, 1e-3, 2e-3, 3e-3], 1000, 500, 1, (nr_cores=3, rank=48), init_scale=[0.9, 1.0, 1.1], id=id) #           10
    collect_measurements(tt, 1000, 4, [7e-4, 1e-3, 2e-3, 3e-3], 1000, 500, 1, (nr_cores=3, rank=64), init_scale=[0.9, 1.0, 1.1], id=id) #           10
    collect_measurements(tt, 1000, 4, [7e-4, 1e-3, 2e-3, 3e-3], 1000, 500, 1, (nr_cores=3, rank=80), init_scale=[0.9, 1.0, 1.1], id=id) #           10
    collect_measurements(tt, 1000, 4, [7e-4, 1e-3, 2e-3, 3e-3], 1000, 500, 1, (nr_cores=3, rank=96), init_scale=[0.9, 1.0, 1.1], id=id) #           10

    collect_measurements(tt, 1024, 4, [7e-4, 1e-3, 2e-3, 3e-3], 1000, 500, 1, (nr_cores=5, rank=64), init_scale=[0.9, 1.0, 1.1], id=id) #           10
    collect_measurements(tt, 1024, 4, [7e-4, 1e-3, 2e-3, 3e-3], 1000, 500, 1, (nr_cores=5, rank=96), init_scale=[0.9, 1.0, 1.1], id=id) #           10
    collect_measurements(tt, 1024, 4, [7e-4, 1e-3, 2e-3],       1000, 500, 1, (nr_cores=5, rank=128), init_scale=[0.9, 1.0, 1.1], id=id) #           10
 
end





begin 
    epochs = 200
    id = 19
    collect_measurements(vit_monarch, 6*64, 6, 5e-4, 128, epochs, 1, (patch_size=4, nr_heads=6, nr_blocks=3),  max_bs=500, weight_decay=0.01, id=id)
    collect_measurements(vit_monarch, 6*64, 6, 6e-4, 128, epochs, 1, (patch_size=4, nr_heads=6, nr_blocks=4),  max_bs=500, weight_decay=0.01, id=id)
    collect_measurements(vit_monarch, 6*64, 6, 6e-4, 128, epochs, 1, (patch_size=4, nr_heads=6, nr_blocks=6),  max_bs=500, weight_decay=0.01, id=id)
    collect_measurements(vit_monarch, 6*64, 6, 7e-4, 128, epochs, 1, (patch_size=4, nr_heads=6, nr_blocks=8),  max_bs=500, weight_decay=0.01, id=id)
    collect_measurements(vit_monarch, 6*64, 6, 7e-4, 128, epochs, 1, (patch_size=4, nr_heads=6, nr_blocks=12), max_bs=500, weight_decay=0.01, id=id)
    # collect_measurements(vit_monarch, 6*64, 6, 8e-4, 128, epochs, 1, (patch_size=4, nr_heads=6, nr_blocks=16), max_bs=500, weight_decay=0.01, id=id)
    collect_measurements(vit_monarch, 6*64, 6, 9e-4, 128, epochs, 1, (patch_size=4, nr_heads=6, nr_blocks=24), max_bs=500, weight_decay=0.01, id=id)
    # collect_measurements(vit_monarch, 6*64, 6, 1e-3, 128, epochs, 1, (patch_size=4, nr_heads=6, nr_blocks=32), max_bs=500, weight_decay=0.01, id=id)
    collect_measurements(vit_monarch, 6*64, 6, 1e-3, 128, epochs, 1, (patch_size=4, nr_heads=6, nr_blocks=64), max_bs=500, weight_decay=0.01, id=id)
end


collect_measurements(vit_dense, 6*64, 6, 5e-4, 128, 200, 1, (patch_size=4, nr_heads=6),  max_bs=500, weight_decay=0.01, id=21)




begin
    id = 40
    nr_runs = 3
    max_epochs = 1200
    wd = 0.0
    lr = 8e-4

    collect_measurements(dense, 1024,  4, lr, 1000, max_epochs, nr_runs, weight_decay=wd, id=id)

    

    
    collect_measurements(lowrank, 1024,  4, lr, 1000, max_epochs, nr_runs, (rank=128,), weight_decay=wd, id=id)
    collect_measurements(lowrank, 1024,  4, lr, 1000, max_epochs, nr_runs, (rank=80,),  weight_decay=wd, id=id)
    collect_measurements(lowrank, 1024,  4, lr, 1000, max_epochs, nr_runs, (rank=32,),  weight_decay=wd, id=id)

    
    collect_measurements(monarch, 1024,  4, lr, 1000, max_epochs, nr_runs, (nr_blocks=16,), weight_decay=wd, id=id)
    collect_measurements(monarch, 1024,  4, lr, 1000, max_epochs, nr_runs, (nr_blocks=32,), weight_decay=wd, id=id)

    collect_measurements(tt, 1024, 4, lr, 1000, max_epochs, nr_runs, [(nr_cores=2, rank=512)], weight_decay=wd, id=id)
    # collect_measurements(tt, 1024, 4, lr, 1000, max_epochs, nr_runs, [(nr_cores=2, rank=384)], weight_decay=wd, id=id)
    collect_measurements(tt, 1024, 4, lr, 1000, max_epochs, nr_runs, [(nr_cores=2, rank=256)], weight_decay=wd, id=id)
    collect_measurements(tt, 1024, 4, lr, 1000, max_epochs, nr_runs, [(nr_cores=2, rank=128)], weight_decay=wd, id=id)
    collect_measurements(tt, 1024, 4, lr, 1000, max_epochs, nr_runs, [(nr_cores=2, rank=80)], weight_decay=wd, id=id)
    collect_measurements(tt, 1024, 4, lr, 1000, max_epochs, nr_runs, [(nr_cores=2, rank=32)], weight_decay=wd, id=id)

    collect_measurements(btt, 1024, 4, lr, 1000, max_epochs, nr_runs, [(nr_cores=2, rank=16)], weight_decay=wd, id=id)
    # collect_measurements(btt, 1024, 4, 1e-3, 1000, max_epochs, nr_runs, [(nr_cores=2, rank=12)], weight_decay=wd, id=id)
    collect_measurements(btt, 1024, 4, lr, 1000, max_epochs, nr_runs, [(nr_cores=2, rank=8)], weight_decay=wd, id=id)
    collect_measurements(btt, 1024, 4, lr, 1000, max_epochs, nr_runs, [(nr_cores=2, rank=4)], weight_decay=wd, id=id)
    collect_measurements(btt, 1024, 4, lr, 1000, max_epochs, nr_runs, [(nr_cores=2, rank=2)], weight_decay=wd, id=id)
    collect_measurements(btt, 1024, 4, lr, 1000, max_epochs, nr_runs, [(nr_cores=2, rank=1)], weight_decay=wd, id=id)


    # collect_measurements(dense, 672,   4, lr, 1000, max_epochs, nr_runs, weight_decay=wd, id=id)
    collect_measurements(dense, 704,   4, lr, 1000, max_epochs, nr_runs, weight_decay=wd, id=id)
    # collect_measurements(dense, 768,   4, lr, 1000, max_epochs, nr_runs, weight_decay=wd, id=id)
    collect_measurements(dense, 832,   4, lr, 1000, max_epochs, nr_runs, weight_decay=wd, id=id)
    # collect_measurements(dense, 896,   4, lr, 1000, max_epochs, nr_runs, weight_decay=wd, id=id)
    # collect_measurements(dense, 960,   4, lr, 1000, max_epochs, nr_runs, weight_decay=wd, id=id)

    id = 41
    max_epochs = 6000
    nr_runs = 1

    collect_measurements(lowrank, 1024,  4, lr, 1000, max_epochs, nr_runs, (rank=512,), weight_decay=wd, id=id)
    # collect_measurements(lowrank, 1024,  4, lr, 1000, max_epochs, nr_runs, (rank=384,), weight_decay=wd, id=id)
    collect_measurements(lowrank, 1024,  4, lr, 1000, max_epochs, nr_runs, (rank=256,), weight_decay=wd, id=id)

    collect_measurements(monarch, 1024,  4, lr, 1000, max_epochs, nr_runs, (nr_blocks=2,), weight_decay=wd, id=id)
    collect_measurements(monarch, 1024,  4, lr, 1000, max_epochs, nr_runs, (nr_blocks=4,), weight_decay=wd, id=id)
    collect_measurements(monarch, 1024,  4, lr, 1000, max_epochs, nr_runs, (nr_blocks=8,), weight_decay=wd, id=id)
end


begin
    collect_measurements(monarch, 1024,  4, 4e-4, 1000, 2500, 3, (nr_blocks=2,), id=32)
    collect_measurements(monarch, 1024,  4, 5e-4, 1000, 2500, 3, (nr_blocks=4,), id=32)
    collect_measurements(monarch, 1024,  4, 6e-4, 1000, 2500, 3, (nr_blocks=8,), id=32)
    collect_measurements(monarch, 1024,  4, 7e-4, 1000, 2500, 3, (nr_blocks=16,), id=32)
    collect_measurements(monarch, 1024,  4, 8e-4, 1000, 2500, 3, (nr_blocks=32,), id=32)

end

begin
    ids = [33, 34, 35, 36, 37]
    nr_epochs = [300, 600, 1000, 1500, 2000]
    for i=1:5
        collect_measurements(monarch, 1024,  4, 6e-4, 1000, nr_epochs[i], 1, (nr_blocks=2,), id=ids[i])
        collect_measurements(lowrank, 1024,  4, 6e-4, 1000, nr_epochs[i], 1, (rank=512,),    id=ids[i])
        collect_measurements(tt     , 1024,  4, 6e-4, 1000, nr_epochs[i], 1, (nr_cores=2, rank=512), id=ids[i])
        collect_measurements(btt,     1024,  4, 6e-4, 1000, nr_epochs[i], 1, (nr_cores=2, rank=16),  id=ids[i])
    end
end

collect_measurements(monarch, 1024,  4, [3e-5, 1e-5], 1000, 300, 1, (nr_blocks=2,), id=27)



begin
    id = 28
    nr_runs = 3
    max_epochs = 10
    wd = 0.01

    collect_measurements(dense, 1024,  4, 7e-4, 1000, max_epochs, nr_runs, weight_decay=wd, id=id)

    collect_measurements(dense, 768,   4, 1e-3, 1000, max_epochs, nr_runs, weight_decay=wd, id=id)


end

begin 
    id=38
    nr_epochs = 2
    collect_measurements(vit_dense, 6*64, 6, 5e-4, 128, nr_epochs, 1, (patch_size=4, nr_heads=6),  max_bs=500, weight_decay=0.01, id=id)
    for rank in [384, 383, 380]
        collect_measurements(vit_lowranklight, 6*64, 6, 5e-4, 128, nr_epochs, 1, (patch_size=4, nr_heads=6, rank=rank),  max_bs=500, weight_decay=0.01, id=id)
    end
end


begin
    id = 50
    wd = [0.0, 0.01]
    collect_measurements(lowranklight, 1024,  4, 1e-3, 1000, 400, 1, (rank=960,), weight_decay=wd, id=id)
    collect_measurements(lowranklight, 1024,  4, 1e-3, 1000, 400, 1, (rank=736,), weight_decay=wd, id=id)
    collect_measurements(lowranklight, 1024,  4, 1e-3, 1000, 400, 1, (rank=512,), weight_decay=wd, id=id)
    collect_measurements(lowranklight, 1024,  4, 1e-3, 1000, 400, 1, (rank=256,), weight_decay=wd, id=id)
    collect_measurements(lowranklight, 1024,  4, 1e-3, 1000, 400, 1, (rank=128,), weight_decay=wd, id=id)
    collect_measurements(lowranklight, 1024,  4, 1e-3, 1000, 400, 1, (rank=64,), weight_decay=wd, id=id)

end

begin 
    id = 102
    nr_epochs = 10
    for bs in [128]
        collect_measurements(vit_dense, 6*64, 6, 5e-4, bs, nr_epochs, 1, (patch_size=4, nr_heads=6),  max_bs=500, weight_decay=0.01, id=id)
        for rank in [64, 128, 192, 256, 320, 380] # 1 to 384
            collect_measurements(vit_lowranklight, 6*64, 6, 5e-4, bs, nr_epochs, 1, (patch_size=4, nr_heads=6, rank=rank),  max_bs=500, weight_decay=0.01, id=id)
        end
    end
end



begin 
    id = 110
    nr_epochs = 40
    bs = 128
    collect_measurements(vit_dense, 6*64, 6, 5e-4, bs, nr_epochs, 1, (patch_size=4, nr_heads=6),  max_bs=500, weight_decay=0.01, id=id)
    collect_measurements(vit_lowranklight, 6*64, 6, 5e-4, bs, nr_epochs, 1, (patch_size=4, nr_heads=6, rank=64),  max_bs=500, weight_decay=0.01, id=id)

end

begin 
    id = 120
    nr_epochs = 10
    collect_measurements(vit_lowranklight, 8*128, 6, 5e-4, 128, nr_epochs, 1, (patch_size=4, nr_heads=8, rank=73),  max_bs=500, weight_decay=0.01, id=id)
    collect_measurements(vit_dense, 6*64, 6, 5e-4, 128, nr_epochs, 1, (patch_size=4, nr_heads=6),  max_bs=500, weight_decay=0.01, id=id)
    
end


begin 
    id = 121
    nr_epochs = 40
    collect_measurements(vit_lowranklight, 8*128, 6, [2.5e-4, 3.75e-4, 5e-4], 128, nr_epochs, 2, (patch_size=4, nr_heads=8, rank=73),  max_bs=500, weight_decay=0.01, id=id)
    collect_measurements(vit_dense, 6*64, 6, [2.5e-4, 3.75e-4, 5e-4], 128, nr_epochs, 2, (patch_size=4, nr_heads=6),  max_bs=500, weight_decay=0.01, id=id)
    
end

begin 
    id = 122
    nr_epochs = 100
    collect_measurements(vit_lowranklight, 8*128, 6, [3.75e-4], 128, nr_epochs, 1, (patch_size=4, nr_heads=8, rank=73),  max_bs=500, weight_decay=0.01, id=id)
    collect_measurements(vit_dense, 6*64, 6, [3.75e-4], 128, nr_epochs, 1, (patch_size=4, nr_heads=6),  max_bs=500, weight_decay=0.01, id=id)

    # collect_measurements(vit_lowranklight, 8*128, 6, 5e-4, 128, nr_epochs, 1, (patch_size=4, nr_heads=8, rank=73),  max_bs=500, weight_decay=0.01, id=id)
    # collect_measurements(vit_dense, 6*64, 6, 5e-4, 128, nr_epochs, 1, (patch_size=4, nr_heads=6),  max_bs=500, weight_decay=0.01, id=id)
    
end

