using LinearAlgebra: norm, dot


# println("Inside polygonal_line_segmenter.jl...")  # DEBUG


function polygonal_line_segmenter(vpp1, vpp2, fp, cutoff)
    # Compute xi1, xi1 tangent and perpendicular
    linevec = vpp2 .- vpp1
    linevecmag = norm(linevec)
    linetan = linevec ./ linevecmag
    xi1 = fp .- vpp1
    xi1tanmag = dot(xi1, linetan)
    xi1perpmag = norm(xi1 .- xi1tanmag * linetan)
    segmidpoint = xi1tanmag * linetan .+ vpp1

    # println("vpp1: ", vpp1)  # DEBUG
    # println("vpp2: ", vpp2)  # DEBUG
    # println("fp: ", fp)  # DEBUG
    # println("cutoff: ", cutoff)  # DEBUG

    # Determine if the field point is before, between or
    # after the segment points, then determine if it is
    # inside the cutoff radius. If it is, slice the segment
    # into parts where the field point is inside the cutoff
    # and outside.
    halfseglen = sqrt(max(0, cutoff^2 - xi1perpmag^2))
    # println("halfseglen: ", halfseglen)  # DEBUG
    # Initialize the return arrays
    incutoff = nothing
    outcutoff = ([vpp1 vpp2],)
    # println("outcutoff: ", outcutoff)  # DEBUG
    if halfseglen > 0
        # (linepoint1)-part1->|segmidpoint|<-part2-(linepoint2)
        part1mag = dot(segmidpoint .- vpp1, linetan)
        part2mag = dot(segmidpoint .- vpp2, linetan)
        seglen1 = min(halfseglen, abs(part2mag))
        seglen2 = min(halfseglen, abs(part1mag))
        tol = 5f-6
        if part1mag < 0
            # Field point before segpoint1
            if norm(xi1) < cutoff
                # Segment
                segpoint1 = vpp1
                segpoint2 = segmidpoint .+ seglen1 .* linetan
                incutoff = [segpoint1 segpoint2]
                if norm(segpoint2 .- vpp2) >= tol
                    outcutoff = ([segpoint2 vpp2],)
                else
                    outcutoff = nothing
                end
            end
        elseif part2mag > 0
            # Field point after segpoint2
            if norm(fp .- vpp2) < cutoff
                segpoint1 = segmidpoint .- seglen2 .* linetan
                segpoint2 = vpp2
                incutoff = [segpoint1 segpoint2]
                if norm(segpoint1 .- vpp1) >= tol
                    outcutoff = ([vpp1 segpoint1],)
                else
                    outcutoff = nothing
                end
            end
        else
            # Field point between segpoint1 and segpoint2
            if xi1perpmag < cutoff
                segpoint1 = segmidpoint .- seglen2 .* linetan
                segpoint2 = segmidpoint .+ seglen1 .* linetan
                incutoff = [segpoint1 segpoint2]
                # println("norm(segpoint2 - segpoint1): ", norm(segpoint2 .- segpoint1))  # DEBUG
                # println("linevecmag: ", linevecmag)  # DEBUG
                if abs(norm(segpoint2 .- segpoint1) - linevecmag) >= tol
                    # println("Inside if statement")  # DEBUG
                    if norm(segpoint1 .- vpp1) >= tol && norm(segpoint2 .- vpp2) >= tol
                        outcutoff = ([vpp1 segpoint1], [segpoint2 vpp2])
                    elseif norm(segpoint1 .- vpp1) >= tol
                        outcutoff = ([vpp1 segpoint1],)
                    else
                        outcutoff = ([segpoint2 vpp2],)
                    end
                else
                    outcutoff = nothing
                end
            end
        end
    end
    # println("incutoff (returned): ", incutoff)  # DEBUG
    # println("outcutoff (returned): ", outcutoff)  # DEBUG
    return incutoff, outcutoff
end


# ####### Test function #######
# using Plots

# # Set up the line segment and field point
# p1 = [0, 0, 0]
# p2 = [5, 0, 0]
# fp = [2, 4.9, 0]
# cutoff = 5

# icf, ocf = polygonal_line_segmenter(p1, p2, fp, cutoff)

# println("Inside cutoff: ", icf)
# println("Outside cutoff: ", ocf)

# plt = scatter([fp[1]], [fp[2]], [fp[3]], label="Field point", markershape=:auto, markercolor=:black)
# if ocf !== nothing
#     for seg in ocf
#         plot!(seg[1, :], seg[2, :], seg[3, :], linecolor=:green, markershape=:x, markercolor=:green, label="Outside cutoff")
#     end
# end
# if icf !== nothing
#     plot!(icf[1, :], icf[2, :], icf[3, :], linecolor=:red, markershape=:circle, markercolor=:red, label="Inside cutoff")
# end

# ylims!(-4, 4)

# display(plt)