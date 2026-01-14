module DPSphericalMeans

using BenchmarkTools
using LinearAlgebra
using MKL

function dp_spherical_means!(
  assignments::Vector{Int},
  centers::Matrix{T},
  counts::Vector{Int},
  X::Matrix{T},
  scores::Vector{T},
  λ::T,
  Kref::Base.RefValue{Int},
  n::Int
)::Nothing where {T<:AbstractFloat}
  d = size(X, 1)

  @inbounds for i in 1:n
    @views xi = X[:, i]
    K = Kref[]

    if K > 0
      @views mul!(scores[1:K], centers[:, 1:K]', xi)
    end

    best_k = 0
    best_score = λ

    @simd for k in 1:K
      s = scores[k]
      if s > best_score
        best_score = s
        best_k = k
      end
    end

    if best_k == 0
      new_k = K + 1

      @simd for j in 1:d
        centers[j, new_k] = xi[j]
      end

      counts[new_k] = 1
      assignments[i] = new_k
      Kref[] = new_k
    else
      c = counts[best_k] + 1
      invc = inv(T(c))

      @simd for j in 1:d
        centers[j, best_k] = invc * (centers[j, best_k] * (c - 1) + xi[j])
      end

      norm2 = zero(T)
      @simd for j in 1:d
        norm2 += centers[j, best_k]^2
      end

      invnorm = inv(sqrt(norm2))
      @simd for j in 1:d
        centers[j, best_k] *= invnorm
      end

      counts[best_k] = c
      assignments[i] = best_k
    end
  end

  return nothing
end

export dp_spherical_means!

end
