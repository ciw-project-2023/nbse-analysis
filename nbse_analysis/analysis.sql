select num_conformers,
       assemblies,
       optimizer_coarse,
       optimizer_fine,
       optimizer_steps,
       divide,
       core
from bench
group by num_conformers, assemblies, optimizer_coarse, optimizer_fine,
         optimizer_steps, divide, core
having count(*) > 1
order by num_conformers asc, assemblies asc, optimizer_coarse asc,
         optimizer_fine asc;


select name, local_similarity, siena_rmsd
from bench
order by siena_rmsd asc;

select name as 'Name', round(avg(bench.local_similarity), 3) as 'Avg. Local Similarity', round(avg(bench.siena_rmsd),3) as 'Avg. SIENA RMSD', round(avg(avg_conformer_tanimoto_dist), 3) as 'Avg. Conformer Score'
from bench
where local_similarity > 0
  and siena_rmsd > 0
group by name
order by avg(local_similarity) desc;




select  count(*),
        num_conformers as 'No. Conformers',
        assemblies as 'Assemblies',
        optimizer_coarse as 'Opt. Coarse',
        optimizer_fine as 'Opt. Fine',
        core as 'Core',
        round(avg(bench.local_similarity), 3) as 'Avg. Local Similarity',
        round(avg(bench.siena_rmsd),3) as 'Avg. SIENA RMSD',
        round(avg(avg_conformer_tanimoto_dist), 3) as 'Avg. Conformer Score',
        round(avg(took), 3) as 'Runtime'
from bench
where core <> 'murcko'
group by num_conformers,
         assemblies,
         optimizer_coarse,
         optimizer_fine,
         optimizer_steps,
         divide,
         core
having count(*) > 1
order by avg(siena_rmsd), avg(local_similarity) desc;


select name, count(*)
from bench
group by name;
