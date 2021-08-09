using HDF5
using Statistics

# ΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛ
# Data Defs
# ΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛ

"""
lattice info stored in this object:
number of spins, energy of interaction,
temperature, spin configuration and energy.
"""
mutable struct Lattice{M <: AbstractArray}
    nspins::Int64
    eps::Float64
    T::Float64
    pbc::Bool
    spins::M
    E::Float64
    M::Float64
    
    # object constructor, throw random spins
    function Lattice(nspins, eps, temp, pbc)
        probs = rand( nspins )
        spins = Vector( [ prob < 0.5 ? -1 : 1 for prob in probs ] )
        
        new{typeof(spins)}(nspins, eps, temp, pbc, spins, 0.0, 0.0)
    end
end

# ΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛ
# Physics Defs
# ΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛ

"""
Calculate the energy (Hamiltonian) of the lattice.
"""
function energy(lat::Lattice)
    nspins = lat.nspins
    spins  = lat.spins
    enrg   = [ spins[i] * spins[i+1] for i in 1:(nspins-1) ]
    if (lat.pbc) 
        append!( enrg, spins[1] * spins[end] )
    end
    lat.E = -lat.eps * sum(enrg)
    return 
end

function magnetization(lat::Lattice)
    return sum(lat.spins)
end

"""
The all-knowing Bolztmann factor
"""
boltz( E, T ) = ( k = 1.38064852e-23; exp( -E/(k*T) ) )

function corr(lat::Lattice)
    if (lat.pbc)
        nspins = lat.nspins
        spins  = lat.spins
        savg   = mean(spins)

        c = ones( (nspins-1)÷2 )
    
        for r in 1:(nspins-1)÷2
            for i in 1:nspins
                ir    = (i + r) == nspins ? nspins : mod( (i + r), nspins ) 
                il    = (i - r) == 0      ? nspins : mod( (i - r), nspins ) 
                c[r] += spins[i]*( spins[ir] + spins[il] )
            end
            c[r] *= 1.0 / ( nspins * 2.0 )
            c[r] -= savg * savg
        end
    end
    return c
end

# ΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛ
# Algorithm Defs
# ΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛ
            
function MC_flip(lat::Lattice, idx)
    nspins = lat.nspins
    spins  = lat.spins
    
    # These are the calculations to obtain ΔE
    # Ef = Ei + eps * ( spins[idx]*spins[idx-1] + spins[idx]*spins[idx+1] ) -
    #           ( (-1)*spins[idx]*spins[idx-1] + (-1)*spins[idx]*spins[idx+1] )
    # Ef = Ei + 2 * ( spins[idx]*spins[idx-1] + spins[idx]*spins[idx+1] )
      
    if lat.pbc
        if idx == nspins
            ΔE = 2 * spins[idx] * ( spins[idx-1] + spins[1] )
        elseif idx == 1
            ΔE = 2 * spins[idx] * ( spins[end]   + spins[idx+1] )
        else
            ΔE = 2 * spins[idx] * ( spins[idx-1] + spins[idx+1] )
        end
    else
        if idx == nspins
            ΔE = 2 * spins[idx] * spins[idx-1]
        elseif idx == 1
            ΔE = 2 * spins[idx] * spins[idx+1]
        else
            ΔE = 2 * spins[idx] * ( spins[idx-1] + spins[idx+1] )
        end
    end
    ΔM = -2 * spins[idx]
    return ΔE, ΔM
end

function MC_step!(lat::Lattice)
    nspins = lat.nspins
    idx    = ceil(Int64, rand() * nspins )

    ΔE, ΔM = MC_flip(lat, idx)
    
    if ΔE <= 0.0
        lat.spins[idx] *= -1
        lat.E += ΔE
        lat.M += ΔM
    else
        if ( rand() < boltz(ΔE, lat.T) )
            lat.spins[idx] *= -1
            lat.E += ΔE
            lat.M += ΔM
        end
    end
    return
end

function MC_run!(iter, lat::Lattice, data)
    
    energy(lat)
    magnetization(lat)
    
    data["mag"][1]     = lat.M
    data["energy"][1]  = lat.E
    data["spins"][:,1] = lat.spins
    
    Eavg = zeros(iter[2])
    
    i = 1
    for step in 1:iter[1]
        MC_step!(lat)
        Eavg[i] = lat.E
        i += 1
        
        if step%iter[2] == 0
            k = step ÷ iter[2]
            println("Step $((k-1)*iter[2])")
            data["energy"][k]  = mean(Eavg)
            data["mag"][k]     = lat.M
            data["spins"][:,k] = lat.spins
            i = 1
        end
    end
    
    data["corr"][:] = corr(lat)
    return
end

# ΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛ
# Interface
# ΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛΛ

function main(args)
    nspins = parse(Float64, args[1]) |> Int #1e4 |> Int
    eps    = parse(Float64, args[2]) #1.0
    temp   = parse(Float64, args[3]) #1.5
    pbc    = true
    iters  = parse(Float64, args[4]) |> Int #1e7 |> Int
    wrt_it = parse(Float64, args[5]) |> Int #1e3 |> Int

    lattice_pbc = Lattice(nspins, eps, temp, pbc)

    data = h5open("./Ising1D.hdf5", "w")
    data_dims = iters ÷ wrt_it 
    create_dataset( data, "energy", datatype(1.0), dataspace( (data_dims,) ) )
    create_dataset( data, "mag",    datatype(1.0), dataspace( (data_dims,) ) )
    create_dataset( data, "corr",   datatype(1.0), dataspace( ((nspins-1) ÷ 2,) ) )
    create_dataset( data, "spins",  datatype(1.0), dataspace( (nspins, data_dims) ) )
    
    
    MC_run!( (iters, wrt_it), lattice_pbc, data )
    
    io = open("julia.out", "w")
    println(io, "Initial Energy = $(data["energy"][1])")
    println(io, "Final Energy = $(data["energy"][end])")
    close(io)

    "Initial Energy = $(data["energy"][1])" |> display
    "Final Energy = $(data["energy"][end])" |> display
    
    close(data)
end

main(ARGS)