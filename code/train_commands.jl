


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
    id = 17

    collect_measurements(btt, 1024, 4, [5e-4, 1e-3, 3e-3],   1000, 500, 1, (nr_cores=2, rank=1), init_scale=[0.7, 1.0, 1.4], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, [5e-4, 1e-3, 3e-3],   1000, 500, 1, (nr_cores=2, rank=2), init_scale=[0.7, 1.0, 1.4], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, [5e-4, 1e-3, 3e-3],   1000, 500, 1, (nr_cores=2, rank=4), init_scale=[0.7, 1.0, 1.4], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, [8e-4, 2.5e-3], 1000, 500, 1, (nr_cores=2, rank=8), init_scale=[0.7, 1.0, 1.4], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, [6e-4, 2e-3],   1000, 500, 1, (nr_cores=2, rank=12), init_scale=[0.7, 1.0, 1.4], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, [5e-4, 1.5e-3], 1000, 500, 1, (nr_cores=2, rank=16), init_scale=[0.7, 1.0, 1.4], id=id) # coresize: 32

    collect_measurements(btt, 1000, 4, [5e-4, 1e-3, 3e-3],   1000, 500, 1, (nr_cores=3, rank=1), init_scale=[0.7, 1.0, 1.4], id=id) # coresize: 32
    collect_measurements(btt, 1000, 4, [5e-4, 1e-3, 3e-3],   1000, 500, 1, (nr_cores=3, rank=2), init_scale=[0.7, 1.0, 1.4], id=id) # coresize: 32
    collect_measurements(btt, 1000, 4, [5e-4, 1e-3, 3e-3],   1000, 500, 1, (nr_cores=3, rank=4), init_scale=[0.7, 1.0, 1.4], id=id) # coresize: 32
    collect_measurements(btt, 1000, 4, [5e-4, 8e-4, 2.5e-3], 1000, 500, 1, (nr_cores=3, rank=8), init_scale=[0.7, 1.0, 1.4], id=id) # coresize: 32

    collect_measurements(btt, 1024, 4, [4e-4, 1e-3, 3e-3],   1000, 500, 1, (nr_cores=5, rank=1), init_scale=[0.7, 1.0, 1.4], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, [4e-4, 1e-3, 3e-3],   1000, 500, 1, (nr_cores=5, rank=2), init_scale=[0.7, 1.0, 1.4], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, [4e-4, 1e-3, 3e-3],   1000, 500, 1, (nr_cores=5, rank=4), init_scale=[0.7, 1.0, 1.4], id=id) # coresize: 32
    collect_measurements(btt, 1024, 4, [2e-4, 8e-4, 2.5e-3], 1000, 500, 1, (nr_cores=5, rank=8), init_scale=[0.7, 1.0, 1.4], id=id) # coresize: 32

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


collect_measurements(vit_tt, 9*64, 6, 4e-4, 200, 500, 1, (patch_size=8, nr_heads=9, nr_cores=2, rank=128), max_bs=200, id=20)